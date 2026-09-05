GO GIVEN ("post after bench"): post on PR #55180 once `swzshapes` has landed and the table below is filled; push f945fa34 (test fold) first.

Thanks — both done or measured:

**2. Tests.** Dropped the two dedicated test functions and the sliced reference (−103 lines). `test_cutlass_fp8_blockwise_scale_gemm` gains two cases, 8193×16384×2560 and 5120×5120×5120, which land on the swizzled path on a 24 MiB-L2 part (and the odd M on the non-swap-AB dispatch) and run the default order elsewhere. Correctness stays CUTLASS's, as you say.

**1. The heuristic, sm120, and more shapes.** I have no sm120 card, so here is what I can say and what I measured:

- On sm120 the gate is a no-op by construction: `weight_bytes > l2CacheSize` never holds for a weight this kernel sees on a 96–128 MiB L2 (the largest in the table below is 128 MiB of FP8 only at 32768×4096, which a 5090 holds and a PRO 6000 holds too), so the launch is bit-for-bit the stock one there. Where a future weight did exceed a 96 MiB L2, swizzle 8 is torch.compile's default for Triton as you note. The activation-slab term exists only because on GB10 the swizzled order costs ~5 % at M = 4096 (166 → 153 TF at 16384×2560) while gaining 3× at M ≥ 8192; it keeps the default order for small chunks. If you would rather have the simpler `weight > L2` rule and accept the small-M loss on GB10, that is a two-line change and I will make it.
- N/K sweep on GB10, swizzle 8 vs 1, same op, bit-identity checked on every cell (`Y` = identical):

[SWEEP TABLE — fill from swzshapes.txt: shape, weight MiB, M, stock TF, sw8 TF, ratio, same]

[SUMMARY SENTENCE — where the gain starts (M, weight), the worst cell for the default order, and any cell where sw8 loses at small M]

Data and harness: `notes/data/swzshapes.txt`, `tools/` in https://github.com/jschmied/qwen38-flash-next-gb10 (same `_C_swz` experiment op as the earlier sweep).
