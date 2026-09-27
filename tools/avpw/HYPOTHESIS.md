FNAVPW adaptive verification at c=1 (2026-09-27 ~08:xx), written before the run. Arms (FN_SEQS=1, capture sizes 1..6,
codeprobe_c1: code greedy / code sampled / prose greedy, 3 starts):
 k3   = fixed K=3, greedy drafts (local argmax)
 k5   = fixed K=5 (block 1728), greedy drafts (local argmax)
 k5av = K=5 drafts, adaptive verify budget (vLLM AdaptiveVerificationManager under PIECEWISE via FN_AV_PW), greedy
        drafts with logits from the NVFP4 slice (FN_DRAFT_PROB=1, local argmax off)
Mechanism: drafts are always 5; the verify batch is trimmed to the budget that maximises expected accepted tokens / ms
(budget from the previous block's confidences). Trimming saves verify rows (~3.3 ms each), not the 2 extra draft steps.
H1 prose: k5 ~+5 % vs k3 (5l); k5av recovers the verify-row part: +1 ... +3 % vs k3 (2 extra draft steps remain).
   drafts_per_step on prose < 5 (the trim is visible), on code closer to 5.
H1 code greedy: k5av within 0 ... +2 % of k5 (small trims + AV overhead: one host sync/step for the confidence copy).
Output: greedy text may differ from fixed K=5 where verify shapes differ (RecoverSSM commit grouping, 5b) -> report
   agreement; timing is the metric.
H0: no prose recovery (budget stays ~5) or code loses > 3 % -> not worth pursuing without FULL graphs / c>1 QSA fix.
