## Purpose

On SM 12.x parts whose L2 does not hold the FP8 weight operand (GB10 / DGX Spark: 24 MiB L2), the CUTLASS blockwise-FP8 GEMM (`CutlassFp8BlockScaledMMKernel`, 128×128 weight blocks, 1×128 activation scales) loses most of its throughput once M spans many rows of tiles: the default CTA raster revisits each weight tile too far apart and the weight is re-streamed from DRAM. Measured on GB10 (`torch.cuda.Event` timing, 5×10 launches, median):

| weight (N×K, FP8) | M=4096 | 8192 | 16384 | 32768 |
| --- | --- | --- | --- | --- |
| 16384×2560 (42 MB), default raster | 165–170 TFLOPS | 86–96 | 52 | 52 |
| 5120×5120 (25 MiB), default raster | 117 | 74 | 73 | 74 |

This is a prefill-time cost on every FP8-blockwise model served on these parts (8k–32k-token prompts).

## Changes

`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8{.cu,_dispatch.cuh}`: when the weight (N·K bytes) exceeds the device L2 (`cudaDeviceProp::l2CacheSize`, read once) and the activation slab (M·K bytes) is at least 14 MiB, the launch sets the persistent tile scheduler's `max_swizzle_size = 8` through the `TileSchedulerArguments` that `cutlass_gemm_caller` already accepts. The swizzled raster visits the M tiles of one weight column group before moving on, so weight tiles are re-read from L2. The result is bit-identical to the default order (each output tile's K-reduction is unchanged). Everything else is untouched; parts whose L2 holds the weight (RTX PRO 6000 Blackwell / GB202, 96–128 MiB) keep the default order, as do launches whose activation slab is below 14 MiB, where the default order is still marginally faster on GB10 (16384×2560 at M=4096: 165–170 vs 149–155 TFLOPS, at M=5120: 166 vs 153) — while 5120×5120 at M=4096 (20 MiB of A) already collapses without the swizzle (117 vs 160). The threshold is empirical, from the table below.

The first revision of this PR chunked M into separate launches instead; reviewer @gau-nernst suggested a proper raster, and the scheduler argument turns out to give the same recovery with none of the machinery (no chunk loop, no scale re-layout, no size heuristic).

## Measurements (GB10, CUTLASS v4.7.1, two starts, all bit-identical to the default order)

| weight | M | default (swizzle 1) | swizzle 2 | swizzle 4 | **swizzle 8 (this PR)** | chunked (rev. 1) |
| --- | --- | --- | --- | --- | --- | --- |
| 16384×2560 | 4096 (A 10 MiB) | 165–170 | 154–164 | 153–162 | 149–155 (gate keeps default) | 151–161 |
| | 6144 | 134–149 | 142–144 | 147–151 | **148–152** | 146–152 |
| | 8192 | 86–96 | 111 | 139–142 | **148–154** | 150–157 |
| | 16384 | 52 | 93 | 142–144 | **152–156** | 151–156 |
| | 32768 | 52 | 94 | 143 | **154** | 156 |
| 10240×2560 | 8192 / 16384 / 32768 | 96 / 63 / 64 | 113 / 95 / 95 | 141 / 140 / 140 | **151 / 150 / 152** | 150 / 151 / 151 |
| 5120×5120 | 4096 (A 20 MiB) | 112–117 | 142 | 153 | **160** | 155 |
| | 6144 | 73 | 122 | 151 | **164** | 152 |
| | 8192 / 16384 / 32768 | 74 / 73 / 74 | 122 / 123 / 122 | 150 / 151 / 150 | **164 / 162 / 163** | 153 / 155 / 155 |

Raster order (`Heuristic` / `AlongM` / `AlongN`) is within noise at swizzle 8; the heuristic is kept.

## Test Plan

- `tests/kernels/quantization/test_cutlass_scaled_mm.py::test_cutlass_fp8_blockwise_large_m`: M ∈ {4096, 4097, 8193, 12288} × weights of 5 MB (control), 25 MiB and 42 MB, plus one 147 MB weight (SM 12.x only): against the dequantized fp32 baseline, and — on SM 12.x — bit-identical to the same GEMM issued as balanced ≤4096-row slices in the default order (balanced so every slice stays in the same kernel configuration).
- `::test_cutlass_fp8_blockwise_compiled_dynamic_m`: one `torch.compile` graph with symbolic M serves both sides of the threshold (frame_count stays 1) and matches eager.
- Standalone build of this exact dispatch on GB10: bit-identity across 3 weights × 10 M values (64…16384) vs the unmodified kernel, timings above.

## Test Result

GB10 (sm_121), CUDA 13.0: all of the above pass; the bit-identity sweep is 30/30.

---

AI assistance: the measurement harness, drafts and this description were produced with Claude Code (Claude Fable 5.1); every line of the change was reviewed by the author.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
