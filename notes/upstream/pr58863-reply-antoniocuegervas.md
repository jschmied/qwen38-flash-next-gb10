POSTED 2026-09-29 (user: "post reply and offer full graph as addition"). vllm-project/vllm#58863, reply to @antoniocuegervas's second-GB10 data point (2026-09-29).

Thanks, this is the first measurement of the PR under real agent load, and the 16-session result is the case it was
written for: the same sessions in less of the pool, so the eviction cliff moves out.

Your reading of the one-stream cost matches ours. On our GB10 (TP=1, same model, MTP K=5, prod config, 2 starts per
arm) we also ran RecoverSSM under PIECEWISE, and we have since added FULL-graph support for the verify path: the GDN
and PLE RecoverSSM metadata builders declare `UNIFORM_BATCH`, and graph padding rows get a null state slot and a
zero-length window, so the verify kernel writes nothing for them. FULL_AND_PIECEWISE vs PIECEWISE, both with
`--use-replayssm`:

| | FULL_AND_PIECEWISE, s1 / s2 | PIECEWISE, s1 / s2 |
|---|---|---|
| code, c=1, greedy (ms/tok) | 14.723 / 14.695 | 15.003 / 14.882 |
| code, c=4 (tok/s) | 143.56 / 142.86 | 142.70 / 143.51 |

That is −1.1…−2.1 % at one stream, level at four, identical greedy text, and it passes the GDN/PLE tests plus a new
padded-row test (91 passed). It should close most of the ~3 % you see against native FULL_DECODE_ONLY.

I'd like to add it to this PR as a follow-up commit, so `--use-replayssm` works with the default FULL_AND_PIECEWISE
mode instead of falling back to PIECEWISE. Reviewers: if you'd rather keep this PR as is, I'll open it as a separate
PR on top. The patch as it stands (188 lines, applies cleanly to this PR's head `5567cc1b`):
[`tools/f4/f4.diff`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/ef5367ab23ac8a5f1180fc41c7714ed1271eeb18/tools/f4/f4.diff); the A/B behind the
table: [§5z](https://github.com/jschmied/qwen38-flash-next-gb10/blob/ef5367ab23ac8a5f1180fc41c7714ed1271eeb18/notes/speed-of-light.md#L1585).

*Disclosure: drafted with Claude (Anthropic); I reviewed the numbers and the post.*
