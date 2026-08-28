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

## Prior art — we were not first

[hashd1ve/qwen38-flash-next-one-dgx-spark](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark)
published the same diagnosis roughly 16 hours before this note, from a different stack (SGLang):

> "13.5 tok/s without speculation is about 41% of the memory-bandwidth roofline, and the reason is
> counter-intuitive: it isn't the experts. The '6B active' are the NVFP4 part — roughly 1.2 GB
> read per token. But the ~3.5B dense parameters (GDN, QSA, mHC, shared expert, embeddings) are
> still BF16 and are read in full on every token: ~7 GB. So the small half of the model dominates
> the clock."

Their prescription is ours too — *"quantizing the dense parameters to FP8 would take the ceiling
from ~33 to ~55 tok/s. That's a requantization project, not a flag."* They did not execute it.

What this note adds is the **kernel-level confirmation** (67% of GPU time in cuBLAS BF16 GEMV,
from a profile rather than an inference), the **95.5% GPU-busy measurement** that rules out the
competing round-trip explanation, and the **execution** — finding and serving a checkpoint that
fixes it.

[starkweatherdigital](https://github.com/starkweatherdigital/qwen3.8-flash-next-nvfp4-recipe)
independently publishes the identical 10.98 GB figure, but as a size fact rather than a
bottleneck; they quantize the *PLE* instead (to NVFP4, 28.8 GB) and measure 16.8 tok/s eager /
24.6 with MTP k=1.

**An unresolved discrepancy worth flagging rather than papering over.** hashd1ve counts 3.5 B
dense / 7 GB; we count 4.84 B / 9.68 GB. They appear to include embeddings and mHC while
excluding `lm_head` (0.64 B) and the dense `mlp` (0.66 B), which we count. The roofline follows
directly from this number — their ~33 tok/s ceiling against our 24.9 — so it should be reconciled
before either is treated as settled. Ours is derived from the safetensors headers of the specific
checkpoint we serve, and is reproducible from the census below.

**A genuine tension in the two accounts.** hashd1ve describes their stack as *latency*-bound —
"at batch 1 a forward is 48 layers of small kernels with the GPU underutilized". Our profile says
95.5% busy. Their supporting evidence is weak (2.23x scaling at c=4 is the *expected* signature of
bandwidth-bound decode, not evidence against it), but they place themselves at 41% of roofline
where we measure ~87%, and both cannot describe the same regime. Different runtimes, so both may
be locally correct — but anyone reading both should know the claims do not compose.

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

**Their published quality metrics do not describe this build — do not rely on them (and we
briefly did).** `gsm8k_metrics.json` and `aime26_metrics.json` are **byte-identical**
(sha256 `88766f7e…` / `eb4acd8c…`) across lovedheart's FP8 build, `RadixArk`'s plain NVFP4 build,
and lovedheart's *pruned* 512→448 variant — the same `latency_seconds` to ten decimal places. The
`model` field inside them says `qwen38next-nvfp4` / `qwen38next-nvfp4-plefp8`, i.e. they were
measured on the **unquantized-dense** build. A pruned model shipping its unpruned parent's numbers
settles it.

So GSM8K 0.9727 / AIME26 0.9875 are real numbers for *RadixArk*, and say nothing about the FP8
conversion. **This checkpoint has no published quality evidence**, which makes our own measurement
mandatory rather than confirmatory. What it does have is process evidence: `fp8_quant_report.csv`
gives per-tensor `mean_rel_err` of 0.0225–0.0227 across all 156 quantized tensors, and
`conversion_environment.json` pins ModelOpt 0.46.0 / CUDA 13.0 / torch 2.13.0+cu130 — the stack we
run. That is a plausible conversion, not a validated one.

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

## Correction: per-token reads, not total dense params

Our earlier figures (4.84 B / 9.68 GB for RadixArk, 2.17 B / 4.33 GB for lovedheart) counted
**all** dense BF16 params. Three groups are not read on a text-decode token, and including them
inflates the roofline:

| group | B params | read per token? |
|---|---:|---|
| vision tower | 0.449 | **no** — only with an image |
| `embed_tokens` | 0.636 | one row, not the matrix |
| MTP drafter | 0.091 | only when speculating |
| experts left in BF16 | 2.517 | sparse, 10 of 512 |

Corrected per-token BF16 remaining after lovedheart: **1.638 B = 3.28 GB/token**, and the full
budget is 3.28 + 2.67 (FP8 dense) + 1.30 (active NVFP4 experts) = **7.25 GB/token**, a ceiling of
**37.7 tok/s** rather than the 32.9 stated earlier.

This probably accounts for much of the discrepancy with hashd1ve's 3.5 B — a per-token-read count
is the correct one, and ours was not.

## What is left, and whether it is quality-critical

| group | GB/token | share | format | risk |
|---|---:|---:|---|---|
| hyper-connections | 1.28 | 39% | FP8 | **unknown — nobody has done it** |
| `lm_head` | 1.27 | 39% | FP8 | **low — we run an FP8 head in prod** |
| `shared_expert` | 0.47 | 14% | MXFP8 | low — `4p89` does exactly this |
| other dense + PLE proj | 0.26 | 8% | FP8 | small, unexamined |
| vision tower / embed / norms | 0 | — | — | **no speed gain at all** |

**`lm_head` is not quality-critical at FP8 — but it is at NVFP4.** We measured this directly on
the sibling Qwen3.8-27B: a RadixArk **NVFP4** head cost **2.4% worse NLL** (8 of 8 chunks) and was
declined for production, while the **FP8** head we run in prod is loss-neutral. The layer is
precision-sensitive; the *format* is what decides it. That makes this the highest-confidence
lever here, and it is worth 1.27 GB/token.

**The hyper-connections are the real experiment.** `attn_hyper_connection.input_mix_weight_{down,up}`
is a low-rank mixing of residual streams, applied at every one of 48 layers with `hc_count: 4`.
Structurally they behave like routers or gates: small tensors whose error compounds through the
whole depth, which is exactly the class practitioners habitually keep in high precision. Nobody
across 28 published checkpoints has quantized them. They are also the single largest remaining
item. Highest value per unit of unknown — and the one worth spending a careful NLL-divergence
measurement on rather than a smoke test.

**Correction (2026-08-28): `shared_expert` is not uncontested — we misread the field.** An
earlier version of this note said nobody had quantized it. That was a false negative from matching
the substring `shared_expert` against ignore-list entries, which hits `shared_expert_gate` — a
different, much smaller tensor. Verified from headers, at least four checkpoints quantize the 48
main `mlp.shared_expert.{gate,up,down}_proj`: `tcclaviger/…-MXFP4-FP8` (MXFP4,
`weight_packed U8 [2560,320]`), `lvkaokao/…-MXFP4-Mixed-CT-AutoRound` (MXFP8 W8A8-dynamic),
`textclf/…-TQ-4bit`, and `local-inference-lab/…-4p89` (MXFP8 g32), the last predating our work.

None of them publishes a measurement of what it buys, so the *measured* delta is still unclaimed —
but the claim to drop is "nobody has done it". **`lm_head` and the hyper-connections do survive
the recheck**: `lm_head` is BF16 in every checkpoint including those four, and every quant config
excludes `*hyper_connection*` (one has 298 such entries).

**Not worth touching for speed:** the vision tower (0.449 B) is never read during text decode, so
quantizing it buys resident memory and zero throughput; `embed_tokens` contributes one row per
token; norms are 1.2 M params in total and genuinely precision-critical. Quantizing the MTP
drafter would degrade acceptance, costing more than it saves.

### Two combinations worth running

| | GB/token | ceiling | vs lovedheart |
|---|---:|---:|---:|
| lovedheart as-is | 7.25 | 37.7 | — |
| **conservative:** `lm_head` FP8 + `shared_expert` MXFP8 | 5.51 | 49.6 | **1.32x** |
| **full:** + hyper-connections FP8 | 4.23 | 64.6 | **1.71x** |

The conservative pair carries evidence for both halves and needs no new science. The full
combination is the one that would put a single Spark past every published number, and it hinges
entirely on whether the hyper-connections tolerate FP8 — which is a measurable question, not a
matter of opinion.

Format choice matters more than layer choice: FP8 sits **2.3%** from BF16 by mean relative error,
NVFP4 **12.1%** — 4.5x further — measured on this box in earlier work.

## The rest of the field, and the floor

A census of 28 Flash-Next checkpoints (safetensors headers over HTTP range reads, nothing
downloaded), ranked by dense GB/token:

| repo | GiB | dense B | GB/tok | PLE | verdict |
|---|---:|---:|---:|---|---|
| `local-inference-lab/…-NVFP4-4p89` | 102.4 | **1.82** | **3.65** | NVFP4 | **unloadable** — see below |
| `lovedheart/…-FP8-Pruned-RTXPRO-6000` | 114.9 | 2.08 | 4.16 | FP8 | experts pruned 512→448 |
| **`lovedheart/…-NVFP4-FP8`** | 123.4 | 2.09 | 4.17 | FP8 | **what we use** |
| `primitive-ai/…-mixed-NVFP4-FP8` | 171.1 | 2.09 | 4.18 | BF16 | too large |
| `RadixArk/…-NVFP4` | 125.9 | 4.76 | 9.52 | FP8 | what we came from |
| `Inferact`, `Qwen/FP8`, `*/W4A16`, `AWQ`, … | 168–174 | 4.76 | 9.52 | BF16 | too large, no gain |
| `Qwen/Qwen3.8-Flash-Next` (BF16 base) | 335.3 | 4.76 | 9.52 | BF16 | — |

Only **one** checkpoint goes further than lovedheart, and it cannot be served. `4p89` quantizes
attention and GDN to MXFP8, experts to NVFP4, and uniquely `shared_expert` to MXFP8 — but it
stores the PLE as NVFP4, `U8 [2500012, 80]` with per-shard `F8_E4M3` scales. vLLM's PLE loader
supports exactly one quantized form, `F8_E4M3 [rows, 160]` plus a single global BF16 scale, and
`ple_layer.py` validates `expected_shape = (rows, 160)`. It fails loudly
(`ValueError: Shape mismatch for PLE embedding shard 0`), not silently — but it fails.

**The floor of the published field is 1.82 B / 3.65 GB per token.** Across all 28 repos nobody
quantizes `lm_head` (0.64 B), the hyper-connection blocks (0.64 B), or the vision tower (0.45 B).
Those plus ~0.1 B of norms are exactly the 1.82 B that `4p89` reaches. Going below it means
quantizing the head or the hyper-connections, which nobody has published.

Also worth knowing: **no checkpoint fits a 128 GB box without PLE offload.** The smallest servable
one is 101.7 GiB, and the PLE table alone is 95.4 GiB at BF16 / 47.7 at FP8 / 26.8 at NVFP4. Any
size comparison that does not say whether the PLE is offloaded is meaningless.

## Why the hyper-connections are unquantized: two blockers, one shared with `lm_head`

They are the largest remaining item — **0.66 B params, 1.32 GB/token**, in 398 tensors — and every
published checkpoint excludes them (one config has 298 `*hyper_connection*` ignore entries). Two
independent reasons, and the first is the same pattern that blocked `lm_head`:

**1. The model hardcodes `quant_config=None`.** `hyperconnection.py` builds them as real
`ReplicatedLinear` / `MergedColumnParallelLinear` modules — so they *could* be quantized — but
passes `quant_config=None` explicitly at three sites (lines 102, 113, 122):

```python
self.input_mix_weight_down = ReplicatedLinear(
    self.hyper_hidden_size, self.lora_rank,
    bias=False, params_dtype=config.params_dtype,
    quant_config=None,                      # <- hardcoded opt-out
    prefix=maybe_prefix(prefix, "input_mix_weight_down"),
)
```

Together with `model.py`'s `ParallelLMHead`, that is **four hardcoded opt-outs** in this model, and
they account for both axes the field has left untouched. A checkpoint can declare these layers all
it likes; the model never asks.

**2. The shapes are not blockwise-FP8 eligible.** They are `(320, 10240)` and `(10240, 320)`, and
`ModelOptFp8PbWoLinearMethod` requires **both** dimensions divisible by 128 — `320 % 128 = 64`. So
the scheme the rest of this checkpoint uses cannot express them at all:

| scheme | 320 divisible? |
|---|---|
| FP8_PB_WO (blockwise 128) | **no** — 320 % 128 = 64 |
| MXFP8 (group 32) | yes |
| NVFP4 (group 16) | yes |
| FP8 per-tensor / per-channel | yes |

So quantizing them is a two-part change: pass `quant_config`, **and** pick a scheme with a group
size that divides 320. MXFP8 is the natural choice — our build already dispatches it, and it is what
`4p89` uses elsewhere. `disable_tp=True` on these layers means TP sharding is not a complication.

Worth noting what this is *not*: `dolf3131`'s skinny-GEMM work targets exactly these shapes
(`(320, 10240)` at M=1, 2.20x, invoked ~97 times per forward) but that is a **kernel** optimisation
on the BF16 weights, orthogonal to quantizing them. Both are available.

### Attempted and failed: MXFP8 hyper-connections on sm_121

We built it. The quantization itself was clean — 194 tensors, round-trip **2.2519%** (the FP8
floor, as predicted), `quant_config` threaded through `GatedResidual` to all three projections,
399 quantized layers, and the offline gate confirmed every axis resolved to `MXFP8` before boot.

It fails at the **kernel**, not the checkpoint:

```
ValueError: Problem size is not supported for mm_mxfp8
  flashinfer/utils.py:1402  <- backend-specific requirement, not the common check
```

`flashinfer`'s `_cutlass_gemm_mxfp8_requirement` has an SM12x branch:

```python
if is_sm12x_supported(a.device):
    # SM120/121 CUTLASS MXFP8 only supports 1D swizzled scales (layout_128x4)
    if use_8x4_sf_layout:                            return False
    if a_descale.ndim != 1 or b_descale.ndim != 1:   return False
    if a.shape[1] % 32 != 0 or b.shape[1] % 32 != 0: return False
```

so on GB10 the MXFP8 path is narrower than on SM100, and the `heuristic_func` falls through to
cudnn or to nothing. vLLM's own loader asserts the checkpoint scale is **2-D unswizzled** and
swizzles it itself, so the checkpoint format we produced is the one vLLM wants — the gap is between
that and what the SM12x CUTLASS kernel will accept for these particular shapes.

**Worth knowing before anyone repeats it:** even had it loaded, the MXFP8 comment in
`process_weights_after_loading` notes that "the emulation kernel may dequant the weight to BF16 at
load time" — an emulation fallback would give **no** bandwidth saving at all, which is the entire
point. Verify which kernel is selected before trusting an MXFP8 win on this hardware.

So the hyper-connections remain unquantized here, now for a third reason on top of the two above:
**no MXFP8 kernel on sm_121 accepts these shapes.** The remaining routes are NVFP4 group-16 (a
different kernel with different constraints, untested) or `dolf3131`'s skinny-GEMM, which speeds up
the same shapes in BF16 and sidesteps quantization entirely.

## What we would do next

1. **Measure lovedheart's quality ourselves** — mandatory now that its published metrics turn out
   to describe a different build.
2. **`VLLM_GDN_DECODE_KERNEL=triton` is required** with FP8 GDN projections: the default CUDA
   kernel deterministically hangs the engine at concurrency ~32, with no error — requests simply
   stall. Reported by `primitive-ai`, who bisected it module class by module class.
3. **Port `4p89`'s MXFP8 `shared_expert` treatment** onto the lovedheart layout — worth ~0.24 B /
   0.47 GB per token, and the MXFP8 dispatch already works in our build. The cheapest remaining win.
4. **Teach `ple_layer.py` the NVFP4 PLE format.** The larger prize: it frees ~21 GiB and unlocks
   `4p89` outright. A real code change — the per-group scale layout is a decode path, not just a
   loader tweak.
5. `--mamba-cache-mode align` — prefix caching is reportedly inert without it, untested here.
