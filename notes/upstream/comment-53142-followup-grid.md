Follow-up with the complete numbers (three server starts, stock stack of the 2026-08-1x preview, EOS-correct 8-turn agent loop, prefix caching on, batch 4096, GB10 sm_121):

| n | acceptance rate (a / b / c) | mean accepted length | s per turn |
| --- | --- | --- | --- |
| 0 | — | — | 2.09 / 1.85 / 1.90 |
| 1 | 72.3 / 72.7 / (pending) % | 1.72 / 1.73 | 2.44 / 2.41 |
| 2 | 64.0 / 65.3 / 60.2 % | 2.28 / 2.31 / 2.20 | 2.40 / 2.19 / 2.12 |
| 3 | 44.8 / 50.2 / 57.9 % | 2.35 / 2.51 / 2.74 | 2.10 / 2.42 / 2.39 |
| 4 | 46.6 / 48.8 / 45.9 % | 2.86 / 2.95 / 2.84 | 2.56 / 2.75 / 2.45 |
| 5 | 37.4 / 40.9 / 40.0 % | 2.87 / 3.05 / 3.00 | 2.48 / 2.18 / 2.36 |
| 6 | 38.5 / 29.7 / 36.9 % | 3.31 / 2.78 / 3.21 | 2.66 / 2.47 / 2.49 |
| 7 | 27.4 / — / 29.4 % | 2.92 / — / 3.06 | 2.88 / — / 2.43 |
| 8 | 25.2 / — / 25.7 % | 3.01 / — / 3.05 | 2.98 / — / 2.78 |

The ladder is unchanged by the block-size fix and by the two other determinism fixes (#54948, the persistent-top-k PR): with those applied the same loop gives 72 / 64 / … % at n = 1 / 2 / … within the start-to-start band above. So this bug is a crash bug, not an acceptance bug — the acceptance "recovery" I first reported here was the `ignore_eos` artifact described in my previous comment.
