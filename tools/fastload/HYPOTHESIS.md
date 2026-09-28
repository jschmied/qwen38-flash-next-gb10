FASTLOAD, 2026-09-28, before the run. User: "take patches 1 and 2", then "all" (early exit, loading, F4).
Arms on the clone venv rssm (1ea7 + overlay), cold page cache per start, 2 starts: stock vs fastload = vllm#58868
(touch mmap'd pages before H2D copies) + blazux patch 16 (expert name index) + blazux patch 18 (MTP name prefilter,
1ea7 port). blazux's own numbers (v0.30, their loader stack incl. 14/15/17): main 450-541 s -> 35.5 s, MTP 46 -> 1.2 s.
H: main weight load -40 ... -70 % (15 and 17 are missing, so not their 35 s); MTP drafter load 62 s -> < 10 s
(prefilter skips ~297k of ~300k tensors); greedy sanity hash identical to stock; the log names the prefilter with
~3.1k kept. Out of range: hash differs (a loader change altered weights) or the MTP load does not drop.
