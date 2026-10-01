On the 0.6.1 port, every resize of the lone stream's graph slot drops its graphs: `_grow` and `_shrink` call
`_state_changed`. The slot starts at 256 rows, grows to 8,192 once a reply's context passes them, and shrinks back when
the request ends. So every lone request recaptures its graphs mid-reply: 23 captures, 3.0 s of a 512-token reply on a
GB10. Two and four streams lose 8–11 % the same way, because the first stream starts alone in the slot.

- The slot keeps its rows when idle. It gives them back only when memory is short: an evicted kept end, the newest
  stream ending, or an idle slot freed for another stream's growth.
- It grows by doubling (at least `STEP` rows), so a long session recaptures a handful of times rather than every request.
- `warm()` captures its graphs last. Before, the warm request grew the slot after the captures, which discarded them.

The two-rank plan is unchanged: `_is_solo` sees through a plan's stand-in, and the new eviction branch is skipped while
planning.

## Measured

One GB10, Flash Next EXL3 3.05bpw, greedy, 512 tokens of code per stream, two starts each:

| | 0.6.0 | `cb5101d` | this PR |
|---|---|---|---|
| `--parallel 4`, one request (ms/token) | 13.63 / 13.52 | 18.41 / 18.56 | **12.18 / 12.20** |
| `--parallel 1` (ms/token) | 12.09 / 12.19 | 12.33 / 12.15 | 12.18 / 12.19 |
| two streams (tok/s) | 111.9 / 111.5 | 99.8 / 99.1 | 111.7 / 110.8 |
| four streams (tok/s) | 158.4 / 157.9 | 146.8 / 145.1 | 157.6 / 157.8 |
| graph captures inside the reply (one / two streams) | — | 23 / 8 | 0 / 0 |

Every reply is byte-identical in all three columns.

## Tests

`tests/test_cuda_growing_caches.py`:
- The new test checks that the slot keeps its rows across a request's end and gives them to another stream when the
  gate is short. It also checks that the slot grows 8,192 → 16,384 → 32,768 → the window.
- The file's decoder helper now sets the port's new attributes (`solo`, `solo_on`, `planning`). Its three existing
  decoder tests failed on `cb5101d` with `AttributeError: ... 'solo'` and pass again.
- The file passes 7 of 7.

I ran the 14 multi-decoder host test files on both commits. This PR fails 29, `cb5101d` fails 32, and the 29 are a
subset of the 32. The remainder fail on this machine for other reasons (mostly `mlx` not installed).

Not covered: int8/int4 KV, the two-rank run, and a server under memory pressure with the slot holding its rows.

Written with AI assistance (Claude Code); the author reviewed every change.
