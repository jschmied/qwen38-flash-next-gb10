POSTED 2026-10-02. TensorFold PR #250 comment (2026-10-02).
Measured on GB10 (sm_121), Flash Next EXL3 3.05 bpw, this branch, one stream, 512 tokens, code and prose prompts, two runs per cell (repeat within 0.6 %). Every arm produced identical tokens.

| | code ms/token | prose ms/token |
|---|---|---|
| default (6 drafts, 0.70) | 12.15 | 18.54 |
| `--mtp-cost 0.06` | 12.12 | 18.53 |
| `--mtp-drafts 8 --mtp-confidence 0 --mtp-cost 0.06` | 12.95 (+6.5 %) | 20.42 (+10 %) |
| `--mtp-drafts 8 --mtp-confidence 0 --mtp-cost 0.15` | **11.70 (−3.9 %)** | 18.72 (+0.9 %) |

- With the 0.70 cut on, 0.06 never fires: it drafts the same 502 / 449 drafts as the default.
- Without the cut, 0.06 over-drafts. The extra drafts are kept 7–8 % of the time, while the rule admits a draft at a chain product of about 0.19. So the head's product overstates greedy acceptance about 2.5× on this checkpoint.
- At 0.15 the in-round stop does beat the fixed cut on code. The idea holds, and the open part is calibration. Two options: scale the product by the per-depth kept/drafted the engine already counts, or price C from the measured rate.
- Startup prices were close: verify 26.0 ms at one row plus ~1.95 ms per row, a draft 1.13 ms. A live round was ~5.6 ms longer than the table, which is host time the rule does not price.

Data: https://github.com/jschmied/qwen38-flash-next-gb10/blob/2fbd7c5/notes/data/tfreview/cost-arms.txt, https://github.com/jschmied/qwen38-flash-next-gb10/blob/2fbd7c5/notes/data/tfreview/cost2-chain.txt

Written with AI assistance (Claude Code).
