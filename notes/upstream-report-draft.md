# DRAFT — upstream report: FlashInfer CUTLASS NVFP4 MoE is nondeterministic on GB10 (sm_121)

*Status: draft, 2026-09-02. Not posted. Awaiting the MoE backend A/B (`cutlass`, `triton`,
`triton_unfused`) to say whether any backend is a mitigation. Numbers below are all from committed
runs; raw data in `notes/data/`.*

## Summary

On Qwen3.8-Flash-Next (NVFP4, hybrid GDN + MoE, 512 experts) served by vLLM
`0.1.dev20073+g8e685d198` on a GB10 (sm_121), **identical requests produce different logits**.
Layer-by-layer hashing across three identical requests locates it: every module upstream is
bit-identical — hyperconnections, the GDN chain, the router logits, the shared expert — and
**`mlp.experts` (the `FLASHINFER_CUTLASS` NvFp4 MoE) is the first and only module that differs.**

## Evidence chain (each step replicated)

1. **Prefix cache off, one 55-token prefill: bit-identical** logits, within and across server
   starts (6 requests / 2 starts, then 6 independent arms). The forward pass is deterministic.
2. **Decode step 1 already differs** (per-token signatures, n=4 arms), with no speculation, no
   CUDA graphs (`--enforce-eager`), no prefix cache.
3. **GDN recurrent state** (conv + SSM, hashed at the request's slot): layer 0 identical through
   every decode step; layer 1 diverges at decode step 1.
4. **PLE (offloaded n-gram lookup)**: inputs identical, output identical, no async race; its
   *input hidden state* differs — i.e. layer 0's deferred hyperconnection block output differs.
5. **All 18 layer-0 submodules, decode step 1**: identical except `mlp.experts` (and `mlp` after
   it). `mlp.gate` — the router — identical, so routing is excluded.
6. **Prefix cache on**: the 55-token prefill is chunked **52 + 3** by the block-aligned split;
   `mlp.experts` is the first differing module in *both* chunks, everything upstream identical.
7. Not a race: forced sync after the align postprocess, `--no-async-scheduling`, and
   `CUDA_LAUNCH_BLOCKING=1` all leave it diverging. Not the runtime: bf16 GEMM / fp32 reduction /
   SDPA / top-k bit-identical over 5 runs incl. allocator churn.

## Shape dependence (open)

`mlp.experts` is identical at M=55 (cache off) and differs at M=52, 3 and 1. Whether that is a
kernel-config threshold, `num_tokens_padded` rows, or cross-call workspace state is not
established. flashinfer#3957 (nvfp4 MoE, silent OOB write corrupting a later 3-token call,
suspected atomic scatter-add finalize) suggests cross-call state.

## Backends tried (same probe, backend verified from the log)

| `--moe-backend` | prefill | decode |
| --- | --- | --- |
| `flashinfer_cutlass` (auto) | identical | differs |
| `marlin` | differs | differs |
| `humming` | differs | differs |
| `cutlass` (`VLLM_CUTLASS`) | **crashes at init** — `Triton Error [CUDA]: an illegal memory access was encountered` | — |
| `triton`, `triton_unfused` | **rejected** — `not supported for NvFP4 MoE` | — |
| **`emulation`** (dequantised) | **identical** | **identical** |

The selectable NvFP4 set on this build is `cutlass`, `flashinfer_trtllm`, `flashinfer_cutlass`,
`flashinfer_cutedsl`, `flashinfer_b12x`, `marlin`, `humming`, `emulation`. `flashinfer_trtllm` has a
known SM121 silent-garbage bug and `flashinfer_cutedsl`/`b12x` are vetoed on this checkpoint by the
unquantised drafter MoE, so every serving-grade backend is now measured.

`VLLM_BATCH_INVARIANT=1` cannot be used: no mamba/linear-attention backend implements
`supports_batch_invariance()` and 36/48 layers are linear attention.

## Control: emulation is deterministic

With `--moe-backend emulation` (dequantise, plain path) all four tokens are identical across three
requests. Every other component — GDN recurrent state, PLE lookup, hyperconnections, sampling — is
therefore reproducible; the divergence is confined to the NVFP4 MoE kernels.

**Cost of the deterministic mode** (8-turn agent loop, 130 tok/turn, c=1, no speculation, prefix
cache on, backend verified per arm): `flashinfer_cutlass` **43.92 ms/tok** (matches the 12-arm
reference 42.8–47.7); `emulation` **51.38 ms/tok** — **+17% decode**. n=1 for emulation, but
no-spec arms reproduce to ~1.10x. TTFT and higher concurrency not measured. So emulation is a
*serving-viable* deterministic option at modest cost, not merely a reference.

## The fix, and why the documented one looked broken

`cutlass_fused_moe(use_fused_finalize=False)` — FlashInfer's own deterministic finalize — **works
and is deterministic on sm_121** (`DETFIN4`: all 4 tokens identical across 3 requests, backend
`FLASHINFER_CUTLASS` verified). It initially failed at engine init with `Invalid gemm2 profile id:
50` (48 with the autotune sweep skipped) because of a second, independent defect:

**The persistent autotune cache key does not distinguish the non-fused runner.** flashinfer#3367
(0.6.13) made the file-cache key `(custom_op, runner_class, profile, extras)` and added
`get_cache_key_extras()` to `TrtllmGemmRunner` only. The cutlass `MoERunner` has no override, so a
runner built with `use_fused_finalize=False` (20 GEMM2 tactics, ids 20-39) reads cache entries
written by the fused runner (40 tactics, ids 20-59) and is handed an id it must reject. A fresh
cache directory sidesteps it; the proper fix is a `get_cache_key_extras()` on `MoERunner`
returning `use_fused_finalize` and the other tactic-table-changing flags (same shape as #3367).

**Two-part fix, both Python, no C++ / JIT rebuild:**
1. vLLM: pass `use_fused_finalize=not <deterministic flag>` in `flashinfer_cutlass_moe.py`
   (`tools/determinism/fusedfinalize_patch.py`, env-gated on `VLLM_MOE_DET_FINALIZE=1`).
2. FlashInfer: `MoERunner.get_cache_key_extras()` (`tools/determinism/moe_cachekey_patch.py`).

Validation of the complete fix against the shared default cache, in all three serving shapes, and
its cost, is running (`DETFIN5`).

## Consequences observed

- MTP acceptance flips per turn between ~89% and ~27% of draft work kept (r = −0.964 with
  ms/tok); no-spec decode is flat (CV 0.05). The drafter and target disagree because the target
  moved.
- Independently reported on the same hardware/model: vllm#54173.

## Repro

`--no-enable-prefix-caching --enforce-eager`, no speculative config, temperature 0,
`max_tokens=4`, `logprobs` with `top_logprobs=20`; send one fixed prompt three times; compare the
per-token top-k vectors. Token 1 matches, token 2 does not. Tools: `tools/determinism/` in this
repo (`logitprobe.py`, `layerhash_patch.py`, `statehash_patch.py`, `plehash_patch.py`).
