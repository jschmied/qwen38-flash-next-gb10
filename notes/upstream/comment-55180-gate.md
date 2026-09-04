Follow-up pushed (gate + test oracle):

- The gate is now the activation slab (M·K ≥ 12 MiB) plus weight > L2, not M > 4096. The sweep has the point that decides it: 5120×5120 at M=4096 (20 MiB of A) is 112–117 TFLOPS in the default order and 160 with swizzle 8, while 16384×2560 at M=4096 (10 MiB of A) is the other way round (165–170 vs 149–155); at M=6144 (15 MiB) every shape gains. Empirical threshold, documented next to the numbers.
- The exact-equality reference now slices M into balanced ≤4096-row launches (4097 → 2049 + 2048, 8193 → 3×2731), so no slice falls into the M ≤ 256 / M ≤ 64 kernel configurations — the test now asserts only what the swizzle guarantees.
- Raster comment softened: the scheduler groups nearby M/N tiles; the traversal order is its heuristic, and AlongM/AlongN measured within noise of it.
