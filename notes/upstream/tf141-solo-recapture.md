One finding on the 0.6.1 port (`cb5101d`, one GB10): every resize of the solo slot drops its graphs (`_grow`/`_shrink` → `_state_changed`). The slot starts at 256 rows and shrinks after each request, so a lone request on `--parallel 4` recaptures 23 graphs mid-reply. That makes it 18.4 ms/token, against 12.2 on `--parallel 1` and 13.6 on 0.6.0. Two and four streams lose 8–11 %. Replies are byte-identical.

#180 is a fix against `pr-141-0.6.1`: the slot keeps its rows until memory is short, and grows by doubling. With it, no captures happen during replies, and every case is back to `--parallel 1` / 0.6.0 speed. Numbers and tests are in the PR.

Written with AI assistance (Claude Code).
