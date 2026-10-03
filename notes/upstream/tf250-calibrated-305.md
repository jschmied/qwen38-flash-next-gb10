DRAFT — needs the user's go. TensorFold PR #250 reply to grearjake-star (2026-10-03).
Reran on `e6ade5e`: EXL3 3.05, GB10, one stream, 512 tokens, one process per arm, so the calibration starts fresh each run. Two rounds; every arm produced identical tokens.

| | code ms/token | prose ms/token |
|---|---|---|
| defaults (6, 0.70) | 12.12 / 12.18 | 18.57 / 18.55 |
| `--mtp-drafts 8 --mtp-confidence 0 --mtp-cost 0.06`, calibrated | 12.67 / 12.41 (+2.0…+4.5 %) | 18.95 / 18.98 (+2.2 %) |
| same with `--mtp-cost 0.15`, calibrated | **11.67 / 11.71 (−3.8 %)** | 18.55 / 18.44 (0…−0.6 %) |

Uncalibrated, the same arms were +6.5 % / +10 % at 0.06 and −3.9 % / +0.9 % at 0.15. So calibration halves the 0.06 loss but doesn't reach the hand-set value, at least within a 512-token run. At 0.15 it is now no worse than the defaults on either prompt type, and 3.8 % faster on code.

So on 3.05 the stop does pay, but only with a C that the checkpoint has to pick. If the flag stays, a per-checkpoint C, or a default derived from the measured round rate, would be the useful part. On 4.05 you see the cut win, so closing it in favour of the cut is also reasonable.

Written with AI assistance (Claude Code).
