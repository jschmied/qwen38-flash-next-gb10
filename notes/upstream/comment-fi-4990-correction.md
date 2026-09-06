# DRAFT — follow-up on flashinfer-ai/flashinfer#4990 (post on go)

Correction and a sharper ask, after more measurement on the same box.

**GEMM1 is at the DRAM floor, not latency-bound.** At M=7503 (512 experts, top-10, 2560→1280) it moves 0.84 GB of
NVFP4 expert weights + 0.10 GB of fp4 activations + 0.19 GB of bf16 output = 1.13 GB per layer; on GB10's 273 GB/s that
is 4.13 ms, and the kernel takes 4.19 ms. The "32 % DRAM throughput" in my ncu table is against a peak ncu assumes for
this part, not the LPDDR5X figure; the `sleeping` (mbarrier) stalls are consumers waiting on bandwidth-bound TMA loads.
Consistent with that, every scheduling change is null or negative: all tactics ±2 %, cold vs warm L2 identical,
`max_swizzle_size` 4/8 −14 % at 29k, and a scheduler permutation that gives each CTA 16 consecutive tiles (bit-identical
output) is −29 % at 7.5k / −26 % at 29k — the strided order lets adjacent CTAs share an expert's weight tiles in L2, which
matters more than the per-tile tensormap switch. So please disregard the occupancy ask.

**Where the time actually is** (torch profiler, one `cutlass_fused_moe` call, NVFP4 W4A4, M=7503; 29263 in parentheses):

| kernel | µs |
| --- | --- |
| grouped GEMM ×2 | 9,994 (21,638): GEMM1 4.19 ms at its 4.13 ms floor; GEMM2 5.6 ms vs a 3.0 ms floor |
| `finalizeMoeRoutingKernel` | 1,861 (7,150) |
| `doActivationKernel` | 1,156 (4,617) |
| `expandInputRowsKernel` | 992 (4,692) |
| routing / prefix sums / strides | ~250 (~960) |
| total | 14,251 (39,058) |

**Two concrete asks for SM120/SM121:**
1. `use_fused_finalize=True` has no effect: the kernel list and times are identical to `False` (14,251 vs 14,143 µs).
   `mayHaveFinalizeFused()` returns true for sm ≥ 90, but the SM120 block-scaled path never runs the FINALIZE epilogue,
   so GEMM2 writes 0.38 GB of bf16 that `finalizeMoeRoutingKernel` reads straight back (1.9 ms per layer at 7.5k).
2. The SwiGLU + fp4 re-quant (`doActivationKernel`) likewise reads GEMM1's 0.19 GB bf16 output back from DRAM.

Fusing those two epilogues would remove roughly 1.1 GB of traffic per layer, about 3 of the 14 ms. GEMM2's remaining
gap to its floor (K=640, five 128-wide K iterations per tile, bf16 output) is the other 2.5 ms; I have not found a
config-level lever for it.
