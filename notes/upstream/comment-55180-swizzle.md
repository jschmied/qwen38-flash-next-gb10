You were right, and it is reachable through the high-level API: `cutlass_gemm_caller` already takes `TileSchedulerArguments`, so the experiment was one argument. On GB10, `max_swizzle_size = 8` with the scheduler's default raster order gives the same recovery as the chunking, bit-identical to the default order (two starts, standalone build of this PR's dispatch with the argument exposed):

| 16384×2560 (42 MB) | M=4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| default order (swizzle 1) | 165–170 TF | 86–96 | 52 | 52 |
| swizzle 2 / 4 | 154–164 / 153–162 | 111 / 139–142 | 93 / 142–144 | 94 / 143 |
| **swizzle 8** | 149–155 | **148–154** | **152–156** | **154** |
| chunked (previous revision) | 151–161 | 150–157 | 151–156 | 156 |

5120×5120 goes further (74 → 163–168 TF at M ≥ 6144, chunking gave 153–155); raster order AlongM/AlongN is within noise of the heuristic. The only cost is at M ≤ 4096 while the weight still fits the L2 (155 vs 170 on the widest weight), so the gate stays the same as before (weight bytes > `l2CacheSize`, plus M > 4096) and everything else goes: the chunk loop, the scale re-layout, the chunk-size heuristic, the chunked-reference test. Pushing the rewrite now; the tests keep the exact-equality check against row-sliced default-order launches, which the swizzled order must match bit for bit and does.

Two things this suggests beyond the PR: CUTLASS's default of `max_swizzle_size = 1` is the wrong default for any part whose L2 is smaller than its weights (swizzle 2 already halves the loss), and the same argument is available for the other CUTLASS launches that go through `cutlass_gemm_caller` (the SM90/SM100 blockwise paths on parts with small L2s).
