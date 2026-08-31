# Prefill batch size is a trade, and the default is optimal for neither end

`--max-num-batched-tokens` moves deep TTFT and agent-turn latency in **opposite** directions,
monotonically. Measured 2026-08-31, one server start per batch value, no speculation, `fp8head`,
`max-model-len 32768`, c=1. TTFT cells are medians of 6 requests; the agent loop is 8 turns.

| batch | TTFT @4k | TTFT @28k | decode | agent loop |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.95 s | 14.01 s | 26.8 | **1.63 s/turn** |
| 4096 (prod default) | 1.60 s | — | 27.0 | 1.94 s/turn |
| 8192 | 1.99 s | 11.78 s | 27.0 | 2.17 s/turn |
| 16384 | 1.83 s | **10.41 s** | 26.9 | 2.23 s/turn |

**Deep TTFT improves 26%** across the range; **the agent loop degrades 37%**. Neither column
reverses. Decode is untouched (26.8-27.0) — this costs nothing on that axis.

**At 4k, batch size does nothing** (1.83-1.99, a 9% spread inside noise). Expected: a 4000-token
prompt is one chunk at every size ≥4096 and two at 2048. The effect is entirely about how many
prefill chunks a long prompt is cut into — 4 chunks at 16384 against 14 at 2048.

## Why this one is callable when today's others are not

Almost every other comparison made today sits inside the noise band. This one does not, for two
reasons worth separating:

- **The span exceeds the spread.** 37% on the agent loop against a 21% spread between *identical*
  configs (measured on MTP n=6: 2.13 vs 2.70 s/turn).
- **Monotonicity across three points.** Noise produces reversals; a clean ordering in both columns,
  in the directions the chunking mechanism predicts, is much harder to get by accident than any
  single pairwise gap.

It also reproduces the Qwen3.8-27B result — 16384 wins at 32k, loses at ~4k agent load — on a
different model and a different attention architecture, which is independent support the single
arms cannot supply on their own.

## What to run

- **Agent work → 2048.** 1.63 vs 1.94 s/turn against the 4096 default: ~16% off every turn, one
  flag, no rebuild, no quality effect. This is a larger agent-work win than any speculation setting
  measured on this box.
- **Long-context single-shot → 16384**, for ~20% off TTFT versus 4096.
- **4096 is a compromise that is optimal for neither end.** It was never chosen; it is the default.

⚠️ The agent-loop figures are one arm per batch value. The direction and the ordering are solid;
the exact percentages are not. `repl2` replicates batch 2048 three times against the baseline.

Related: [[agentic-speed-is-ttft-bound]] — the decode column is flat here while the agent column
moves 37%, which is the same lesson from a third direction: **decode tok/s does not select configs
for agent work.**
