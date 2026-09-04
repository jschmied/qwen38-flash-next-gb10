Ran both of your kernels on the GB10 at this PR's shapes (standalone, two starts, all bit-identical to CUTLASS; full table in [finding 99](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/prefill-investigation.md)):

| 16384×2560 FP8 weight (40 MiB) | M=4096 | M=8192 | M=16384 |
|---|---|---|---|
| CUTLASS `sm120_fp8_blockwise`, single launch | 164 TF | 92 | 52 |
| your Triton `_fp8_1d2d_kernel` (BM128/BN64, swizzle 8) | 101 | 101 | 101 |
| your CuteDSL `sm120_mm_fp8_1d2d` | 139 | 97 | 50 |
| this PR (C++ chunking, same CUTLASS mainloop) | 155–170 at every M | | |

Same picture on 10240×2560, 12288×2560 and 5120×5120. So on GB10 the two kernels split the problem: the Triton kernel's `swizzle2d` raster is exactly what the 24 MiB L2 needs (flat over M, 1.9× the collapsed CUTLASS at 16k) but Triton's FP8 mainloop caps at ~100 TF on sm_121 (`dot_scaled` is slower here, 66 TF, so not worth gating on); the CuteDSL kernel has the mainloop (0.84× CUTLASS at 4k) but no L2-aware raster and collapses identically. A CuteDSL kernel with the swizzle would be the proper fix you describe and should land above chunking; until one exists, chunking keeps the mainloop and the numerics. Happy to run a swizzled CuteDSL variant here if you push one — the harness is `tools/gn_fp8_bench.py` in the repo above (needs `CUTE_DSL_ARCH=sm_121a`; cutlass-dsl 4.6.2 imported your kernel fine).
