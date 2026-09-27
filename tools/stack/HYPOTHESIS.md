Stack measurement (2026-09-27 ~05:xx), written before the run. Arms (codeprobe, 2 starts):
 today    = K=3, greedy drafts (local argmax), FP8 GDN      (= current prod-like decode config on the PR code)
 k5prob   = K=5 (block 1728) + probabilistic drafting, FP8 GDN     (both exact levers, 5l + 5n)
 k5probnv = k5prob + NVFP4 W4A16 GDN projections (5j; quality trade NLL +0.4 % vs BF16)
Expected (from 5j/5l/5n, assuming the levers multiply):
 code greedy  ms/tok: today ~16.9, k5prob ~15.2 (-10 %), k5probnv ~14.3 (-15 %)
 code sampled ms/tok: today ~18.0, k5prob ~16.0 (-11 %), k5probnv ~15.1 (-16 %)
 prose greedy ms/tok: today ~23.3, k5prob ~24.4 (+5 %),  k5probnv ~23.0 (-1 %)
Out of range (levers not multiplying by > 3 pp) -> look for an interaction (e.g. NVFP4 GDN lowering K=5 acceptance more
than K=3's, as 5j's accept 2.53 -> 2.49 at K=3).
