# What limits single-stream decode

**Answer: two-thirds of it is cuBLAS BF16 matrix-vector kernels, running on the ~4.84B dense
parameters the RadixArk checkpoint leaves unquantized.** The GPU is busy, not stalled. The fix is
a more completely quantized checkpoint, not a scheduling or CUDA-graph change.

Measured 2026-08-27, torch profiler over a steady-state 120-token decode, 58.8 ms/token
(17.0 tok/s), no speculation.

## GPU busy: 95.5%

Computed as the **union of 253,529 kernel intervals across 52 streams** from the raw trace, which
is the only way to get this right — kernels overlap (the PLE `copy_stream` runs alongside compute),
and the profiler's summary table double-counts, with its parent ranges and child kernels summing
to 158%.

```
trace span                     7.054 s
GPU busy (>=1 kernel running)  6.733 s  = 95.5%
main compute stream alone      6.650 s  = 94.3%
idle gaps                              ~2.6 ms/token
```

> An earlier draft asserted 94.6% busy by reading it off that double-counted percentage column.
> The number survived a proper recomputation, but the derivation was wrong and should not have
> been published as it stood.

**This rules out the round-trip explanation for our stack.** 0xBakeer and paragontasx attribute
~22 ms of a 36 ms step (resp. ~29 of 38) to fixed PLE-path overhead — a CPU gather plus a
*pageable* host-to-device copy forcing a per-token CUDA-graph break. With only 2.6 ms/token of
idle, no such stall exists here. Their measurements are on llama.cpp with mmap-based PLE handling,
a different mechanism from vLLM's offload worker; both can be right about their own stack.

## Where the GPU time goes

| group | ms/token | % of kernel time | calls/token |
|---|---:|---:|---:|
| **cuBLAS GEMV (BF16 mat-vec)** | **40.77** | **66.7%** | 339.4 |
| cutlass GEMM (the NVFP4 path) | 15.80 | 25.8% | 582.8 |
| elementwise / norm | 1.87 | 3.1% | 487.7 |
| other | 1.33 | 2.2% | 492.8 |
| GDN / linear attention | 0.59 | 1.0% | 72.6 |
| MoE routing + experts | 0.47 | 0.8% | 48.4 |
| QSA sparse attention | 0.29 | 0.5% | 62.5 |

339 GEMV calls per token over 48 layers is ~7 per layer — the attention and GDN projections.
The MoE experts, which are properly NVFP4, cost **0.47 ms/token**. The unquantized dense
projections cost **87x that**.

## Why: the checkpoint's dtype census

`RadixArk/Qwen3.8-Flash-Next-NVFP4`, read from the safetensors headers:

```
U8        60.40 B   NVFP4-packed experts
F8_E4M3   58.75 B   PLE table + expert scales
BF16       8.00 B   <- the problem
```

The **dense** part of that BF16 — read in full on every single token:

| | params |
|---|---:|
| `linear_attn` (GDN projections) | 1.14 B |
| `qkv` | 1.05 B |
| `mlp` | 0.66 B |
| `lm_head` | 0.64 B |
| `q_proj` / `o_proj` / `k_proj` / `v_proj` | 0.67 B |
| `shared_expert` (always active) | 0.24 B |
| other | 0.44 B |
| **total dense BF16** | **≈4.84 B** |

At 2 bytes each that is **9.7 GB per token**. GB10 has 273 GB/s, so those weights alone impose a
**35.5 ms/token floor** — 60% of the 58.8 ms budget, from a bandwidth model that knows nothing
about the profile. The profiler independently attributes 40.8 ms/token to GEMV. Two unrelated
methods, same answer.

If those parameters were NVFP4 (~0.55 B/param including scales) they would cost 2.66 GB and
9.7 ms/token, projecting **~29.8 tok/s** unspeculated.

**Correction — the obvious fix is not the Inferact checkpoint.** An earlier version of this note
said switching to `Inferact/Qwen3.8-Flash-Next-NVFP4` was the largest lever, on the strength of
[dolf3131's](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark) parameter accounting. A
remote dtype census (safetensors headers over HTTP range reads, no download) shows Inferact has
**exactly the same 4.84 B of dense BF16 as RadixArk**, group for group:

| group | RadixArk | Inferact |
|---|---|---|
| attn q/k/v/o | BF16 1.72 B | BF16 1.72 B |
| GDN / linear attn | BF16 1.14 B | BF16 1.14 B |
| dense mlp | BF16 0.66 B | BF16 0.66 B |
| lm_head | BF16 0.64 B | BF16 0.64 B |
| PLE table | **F8_E4M3 51.2 B** | **BF16 51.2 B** |
| **dense in BF16** | **4.84 B** | **4.84 B** |

The entire 170.2 vs 125.9 GiB difference is the **PLE precision**, not the dense weights.
Switching would cost 44 GiB and gain nothing on decode. RadixArk is the better of the two for us.

## The checkpoint that does fix it

`lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` (123.5 GiB — slightly *smaller* than RadixArk)
quantizes exactly the layers this profile indicts, and keeps the compact FP8 PLE:

| group | RadixArk | lovedheart |
|---|---|---|
| attn q/k/v/o | BF16 1.72 B | **F8_E4M3 1.54 B** |
| GDN / linear attn | BF16 1.14 B | **F8_E4M3 1.13 B** |
| PLE table | F8_E4M3 51.2 B | F8_E4M3 51.2 B |
| **dense in BF16** | **4.84 B** | **2.17 B** |

Projected effect, using the same bandwidth model and holding the measured 68% efficiency:

| | GB/token | ceiling | projected |
|---|---:|---:|---:|
| RadixArk | 10.98 | 24.9 tok/s | 17.0 measured |
| lovedheart | 8.31 | 32.9 tok/s | **~22.4** (+32%) |

FP8 rather than NVFP4 for these layers is the right call on quality grounds: measured on this box
in earlier work, NVFP4 weights sit **4.5x further from BF16 than FP8** (12.1% vs 2.7% relative
error), and the attention projections are the precision-sensitive part.

Their published evidence: GSM8K **0.9727**, AIME26 pass@1 **0.9875** / majority@8 **1.0**, built
with ModelOpt 0.46.0 on CUDA 13.0 / torch 2.13.0+cu130 — the same stack we run. `hf_quant_config.json`
declares `MIXED_PRECISION` through **`quantized_layers`**, which is the field vLLM actually reads;
declaring it via `config_groups` instead produces a W4A4 kernel with no `input_scale` and silent
zero-length output. All 208 shards publish `lfs.sha256`, so it can be verified properly.

Untested by us. The remaining 2.17 B (lm_head, dense mlp, shared_expert, misc) is what a local
re-quant would still have to address.

## Speculative decoding: measured

In-checkpoint MTP (`mtp_num_hidden_layers: 1`, `hc_count: 4`, 31 MTP tensors — present all along,
we simply never enabled it). `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'`.

| c | no spec | MTP k=2 | MTP k=3 |
|---:|---:|---:|---:|
| 1 | 17.1 | **28.5** (+67%) | 27.4 |
| 2 | 33.4 | **41.3** | 38.7 |
| 4 | 44.1 | 50.7 | **60.6** |
| 8 | 87.5 | 89.0 | **93.4** |
| 16 | 131.6 | 143.7 | — |

Acceptance: k=2 accepts 77.1% / 58.5% at positions 0-1 (mean length 2.36 light, 2.16 under load);
k=3 accepts 70.3% / 48.2% / 33.0% (mean 2.52).

**The optimum k shifts with concurrency** — k=2 wins at c<=2, k=3 wins from c=4. k=3 drafts longer
but costs more per step, and batching absorbs that cost. Every published comparison we have seen
tests single-stream only, where k=2 wins; that result does not transfer to a loaded server.

**Speculation and batching are substitutes, not complements.** MTP is worth +67% at c=1 and
+2..9% at c>=8: both amortize the same weight read, so once the batch is doing it, the drafter has
little left to recover. On a box serving one user, enable MTP; on a loaded one, raise
`--max-num-seqs` first.

## What we would do next

1. **Switch checkpoint.** The 4.84 B of BF16 dense weights is the single largest lever and it is
   somebody else's solved problem — Inferact quantizes more of it.
2. Quantize the PLE too (starkweatherdigital: 102.4 -> 28.8 GB; Death-By-Tokens HashK: -> 12.8 GB),
   freeing memory for KV or a bigger draft head.
3. `--mamba-cache-mode align` — prefix caching is reportedly inert without it, untested here.
