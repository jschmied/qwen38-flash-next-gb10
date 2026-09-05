POSTED 2026-09-05 (see posting log).

Thanks — both done or measured:

**2. Tests.** Dropped the two dedicated test functions and the sliced reference (−103 lines). `test_cutlass_fp8_blockwise_scale_gemm` gains two cases, 8193×16384×2560 and 5120×5120×5120, which land on the swizzled path on a 24 MiB-L2 part (and the odd M on the non-swap-AB dispatch) and run the default order elsewhere. Correctness stays CUTLASS's, as you say.

**1. The heuristic, sm120, and more shapes.** I have no sm120 card, so here is what I can say and what I measured:

- On sm120 the gate is a no-op by construction: `weight_bytes > l2CacheSize` never holds for a weight this kernel sees on a 96–128 MiB L2 (the largest below is 128 MiB of FP8 at 32768×4096, which a PRO 6000 holds), so the launch is bit-for-bit the stock one there; a 5090 (96 MiB) would swizzle only that one shape, where the data below says it is 3× on the part that needs it and swizzle 8 is torch.compile's default anyway.
- N/K sweep on GB10, swizzle 8 vs 1, same op, bit-identity checked on every cell (`Y` = identical):

| N×K | weight | M=2048 | M=4096 | M=6144 | M=8192 | M=12288 | M=16384 |
|---|---|---|---|---|---|---|---|
| 16384x2560 | 40 MiB | 153→172 (×1.12) | 167→153 (×0.92) | 146→153 (×1.04) | 93→152 (×1.63) | 54→156 (×2.92) | 53→156 (×2.93) |
| 12288x2560 | 30 MiB | 153→160 (×1.04) | 160→154 (×0.96) | 107→152 (×1.43) | 93→153 (×1.65) | 53→153 (×2.87) | 53→153 (×2.91) |
| 10240x2560 | 25 MiB | 147→157 (×1.07) | 155→152 (×0.98) | 145→152 (×1.05) | 90→154 (×1.72) | 70→152 (×2.18) | 70→154 (×2.22) |
| 2560x6144 | 15 MiB (fits L2) | 178→163 (×0.92) | 161→160 (×0.99) | 172→165 (×0.96) | 171→164 (×0.96) | 168→166 (×0.99) | 172→168 (×0.97) |
| 5120x5120 | 25 MiB | 164→162 (×0.99) | 123→166 (×1.34) | 76→168 (×2.21) | 86→164 (×1.90) | 82→168 (×2.06) | 77→169 (×2.20) |
| 7168x5120 | 35 MiB | 166→164 (×0.99) | 123→164 (×1.33) | 53→172 (×3.26) | 58→170 (×2.92) | 57→170 (×2.96) | 58→169 (×2.90) |
| 4096x4096 | 16 MiB (fits L2) | 162→163 (×1.01) | 158→161 (×1.01) | 163→163 (×1.00) | 163→164 (×1.00) | 165→164 (×0.99) | 160→164 (×1.03) |
| 8192x8192 | 64 MiB | 158→159 (×1.01) | 72→168 (×2.32) | 54→174 (×3.21) | 53→171 (×3.21) | 54→172 (×3.22) | 54→171 (×3.19) |
| 14336x4096 | 56 MiB | 161→165 (×1.03) | 153→164 (×1.07) | 62→164 (×2.66) | 56→166 (×2.99) | 50→166 (×3.31) | 52→167 (×3.24) |
| 32768x4096 | 128 MiB | 165→167 (×1.01) | 160→162 (×1.02) | 65→165 (×2.54) | 54→166 (×3.06) | 52→167 (×3.23) | 53→168 (×3.20) |

bit-identical everywhere: True

Reading it: the swizzled order is flat at 150–174 TF at every M and every shape; the default order is the erratic one (54 TF at M ≥ 6144 on 8192×8192, 50–53 on 14336×4096 and 32768×4096 from M = 6144–8192). Where the weight exceeds the L2 the swizzle is equal or up to 3.3× faster except one narrow band, M = 4096 on the 2560-wide weights (0.92–0.98) — and that band is the only thing my activation-slab term ever protected, while costing 12 % at M = 2048 on 16384×2560. Where the weight fits the L2 (2560×6144, 4096×4096) the swizzle is neutral to −8 %, which `weight > L2` excludes. So you were right on the heuristic: **the gate is now `weight > L2` only** (af52cd77), and the test change is folded into the existing blockwise test with two prefill-sized cases (f945fa34).

Data and harness: `notes/data/swzshapes.txt`, `tools/` in https://github.com/jschmied/qwen38-flash-next-gb10 (same `_C_swz` experiment op as the earlier sweep).
