README0928 re-measure (2026-09-28 ~16:30, user: "check if values in our README still apply after all patches, if needed
remeasure"). The prod config changed today (bf16 SSM state, pmu 64, fastload, tool guards, HC fusion, F4); most README
rows predate it. One armrun spec on prod's venv + launcher + every drop-in's env (the prodval3 env), KV 4 GiB, 2 starts.
Probe readmeprobe.py = nvprobe (code/prose c=1, sampled, c=4, TTFT 8k/30k, replay) -> concprobe (c=4/8/16 prose) ->
agentloop2 (8 dependent turns) -> turnreplay (46 SWE turns + 20k-prefix cells), merged into one json.
Predicted, against the README rows:
- code c=1 greedy 65.7 tok/s (15.2 ms, fp32 state) -> 14.4..15.0 ms/tok (66.7..69.4 tok/s); sampled 16.1 -> 16.0..16.7
- prose c=1 24.3 ms/tok (41.1 tok/s) -> 23.3..24.3
- code c=4 135.0..138.3 tok/s -> 138..148
- TTFT 8k / 30k 2.77..2.78 / 10.25..10.33 s -> 2.55..2.72 / 9.4..9.9 s (HC fusion)
- agent loop 1.10 s/turn (09-26, K=3) -> 0.9..1.15 s/turn
- warm turns B median 0.57 s -> 0.54..0.59 s
- c=16 prose aggregate (README: "~100/110 at 16/32, previous stack") -> 180..260 tok/s; c=32 not measurable (max-num-seqs 16)
- KV in 4 GiB 105,325 tokens (from the log; F4/HC do not change the fixed pool)
Out of range on any row: record the measured value, do not re-fit the prediction.
