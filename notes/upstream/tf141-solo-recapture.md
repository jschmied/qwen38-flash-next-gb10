DRAFT — needs the user's go. ashhart/TensorFold #141 comment, on the 0.6.1 port (2026-10-01 13:10).

A measurement on the 0.6.1 port (`pr-141-0.6.1` at `cb5101d`), one GPU, before it ships. The solo slot's graphs are
recaptured on every resize: `_grow` and `_shrink` call `_state_changed`, the slot starts at 256 rows and shrinks back
after each request, so a lone request recaptures ~23 graphs mid-reply once its context passes the slot's rows.

GB10, Flash Next EXL3 3.05bpw, greedy, 512 tokens of code, two starts each:

| | 0.6.0 | port | port + fix |
|---|---|---|---|
| `--parallel 4`, one request (ms/token) | 13.63 / 13.52 | **18.41 / 18.56** | 12.18 / 12.20 |
| `--parallel 1` (ms/token) | 12.09 / 12.19 | 12.33 / 12.15 | 12.18 / 12.19 |
| two streams (tok/s) | 111.9 / 111.5 | 99.8 / 99.1 | 111.7 / 110.8 |
| four streams (tok/s) | 158.4 / 157.9 | 146.8 / 145.1 | 157.6 / 157.8 |
| captures inside the reply (one / two streams) | — | 23 (3.0 s) / 8 (0.9 s) | 0 / 0 |

Every reply is byte-identical in all three columns. With the slot already grown before the request, the port takes 0
captures and runs at 12.13 ms/token, so the solo path itself is right.

A fix on top of `cb5101d`, if useful:
[`153817c`](https://github.com/jschmied/TensorFold/commit/153817cba1b6af1de92430f0445d40c0192d8d2d) (`multi.py`,
+15/−5). The solo slot keeps its rows when idle and gives them back only when memory is short. It grows by doubling, so
a long session recaptures a handful of times. `warm()` captures its graphs last, because the warm request grew the slot
after the captures and discarded them. The 14 multi-decoder host test files fail the same set before and after. Not
covered: int8/int4 KV, the two-rank plan (the new eviction branch is skipped while planning), and memory pressure with
the slot holding its rows.

Written with AI assistance (Claude Code).
