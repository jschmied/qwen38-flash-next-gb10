POSTED 2026-10-01 (user: "post"). ashhart/TensorFold #136 comment (2026-10-01).

Some measurements for this, from one GB10 (Flash Next, EXL3 3.05 bpw, TensorFold 0.6.0). The first table is the A/B on
TensorFold itself. The rest is from vLLM on the same model.

`--mtp-drafts` × `--mtp-confidence`, greedy, 512 tokens per reply, two starts each. Replies are byte-identical in every
cell.

| one stream, ms/token | 6 / 0.7 (default) | 6 / 0.8 | 7 / 0.8 | 8 / 0.8 |
|---|---|---|---|---|
| code | 12.16 / 12.22 | 12.27 / 12.27 | 12.06 / 12.08 | 11.99 / 12.07 |
| English prose | 18.63 / 18.63 | 18.70 / 18.77 | 18.74 / 18.72 | 18.75 / 18.76 |

With four streams (code, and English/German/Chinese/French prose), every cell is within ±1 % of the default. Measured
by summed per-stream decode time, because aggregate tok/s follows the slowest stream. At 0.8 the chains rarely reach
the sixth draft, so depth 7 and 8 add almost nothing. On these prompts the default leaves about 1 % for any depth rule.

From vLLM on the same model (replays of logged drafts, and server runs):
- An extra draft costs a near-constant ~4.8 ms (draft step plus verify row).
- Acceptance by position is .89 .76 .64 .55 .45 on code and .70 .46 .30 .19 .12 on prose. The best fixed depth is 5
  for code and 3 for prose.
- Choosing the next round's depth from the last round's confidences lost in replay: −0.8 to −8.2 % on code, +1.9 % at
  best on prose. One round's confidences do not predict the next round's.
- A stop inside the round won. The best τ was 0.70, flat over 0.60–0.75, which is TensorFold's default. Against fixed
  depth 5 it gained +4 % on code and +7.5 % on prose; depth 7 with τ 0.8 gained +10 % and +12.8 %.
- Trimming verify rows alone, leaving the draft steps, lost on prose (+3 % slower). The gain is in the skipped draft
  steps.

So the expected-tokens-per-ms rule looks right as an objective. On this model the stop TensorFold already has captures
most of it. The data, the replays, and the probe that sets these flags are in our notes if you want to rerun them.

Written with AI assistance (Claude Code).
