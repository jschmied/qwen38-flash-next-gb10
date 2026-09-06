# DRAFT — follow-up on flashinfer-ai/flashinfer#4990 (post on go)

Correction and a sharper ask, after more measurement on the same box (GB10, sm_121, FlashInfer 0.6.17 AOT `fused_moe_120`).

**GEMM1 is at the DRAM floor, not latency-bound.** At M=7503 (512 experts, top-10, 2560→1280) it moves 0.84 GB of
NVFP4 expert weights + 0.10 GB of fp4 activations + 0.19 GB of bf16 output = 1.13 GB per layer; at GB10's 273 GB/s that
is 4.13 ms, and the kernel takes 4.19 ms. The "32 % DRAM throughput" in my table is against a peak ncu assumes for this
part, not the LPDDR5X figure, and the `sleeping` (mbarrier) stalls are consumers waiting on bandwidth-bound TMA loads.
Consistent with that, every scheduling change is null or negative: cold vs warm L2 identical, `max_swizzle_size` 4/8 −14 %
at 29k, and a scheduler permutation giving each CTA 16 consecutive tiles (bit-identical output) is −29 % at 7.5k /
−26 % at 29k — the strided order lets adjacent CTAs share an expert's weight tiles in L2. Please disregard the occupancy ask.

**Two things I got wrong in the issue, for the record.** (1) The tactic table is not flat: `cutlass_fused_moe(...,
profile_ids=...)` is accepted but ignored in 0.6.17 (tactics always come from `AutoTuner.choose_one`, and outside
`autotune()` that is the fallback tactic −1), so my "all tactics within ±2 %" was the same fallback run each time. Forcing
tactics through `get_valid_tactics` inside `autotune(True)` shows the autotuner's pick (fused-finalize GEMM2, scatter
epilogue) at 13.47 ms per layer vs 14.07 ms for the fallback (−4.3 %). If `profile_ids` is meant to work, that is a small
bug; if not, the docstring should say so. (2) The fused finalize does run on SM120 once autotuned.

**Per-layer budget with the autotuned tactics, M=7503 (29263 in parentheses):**

| kernel | µs |
| --- | --- |
| GEMM1 (plain epilogue) | 4,190 — at its 4,130 µs DRAM floor |
| GEMM2 with fused finalize (scatter epilogue) | 5,120 (floor 3,030 for the bytes; the scatter's atomics are the rest) |
| `doActivationKernel` (SwiGLU + fp4 requant) | 1,140 (4,620) |
| `expandInputRowsKernel` | 990 (4,690) |
| routing / prefix sums / strides | ~250 (~960) |
| total | 13,470 (~37,000) |

**Remaining ask for SM120/SM121:** fuse the SwiGLU + fp4 re-quantization into GEMM1's epilogue. GEMM1 writes 0.19 GB
of bf16 at 7.5k that `doActivationKernel` reads straight back; the fused-gated-activation path exists for the Ampere
kernels but not for the TMA warp-specialized SM120 path. That is ~1.1 ms of 13.5 per layer here.
