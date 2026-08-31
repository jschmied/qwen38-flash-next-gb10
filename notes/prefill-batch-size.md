# Prefill batch size: the TTFT half holds, the agent-loop half was measured wrong

Measured 2026-08-31, one server start per batch value, no speculation, `fp8head`,
`max-model-len 32768`, c=1. TTFT cells are medians of 6 requests.

| batch | TTFT @4k | TTFT @28k | decode |
| ---: | ---: | ---: | ---: |
| 2048 | 1.95 s | 14.01 s | 26.8 |
| 4096 (prod default) | 1.60–1.68 s | 11.96 s | 27.0 |
| 8192 | 1.99 s | 11.78 s | 27.0 |
| 16384 | 1.83 s | **10.41 s** | 26.9 |

**Deep TTFT improves monotonically, 26% across the range**, and decode is untouched. At 4k batch
size does nothing (1.83–1.99, inside noise) — a 4000-token prompt is one chunk at every size
≥4096. The effect is purely how many chunks a long prompt is cut into: 4 at 16384 against 14 at
2048. This part stands.

The curve is not smooth. The drops are 2048→4096 (−15%) and 8192→16384 (−12%), with 4096→8192
nearly flat (−1.5%). **Our 4096 default already captures most of the available deep-TTFT gain**;
going to 16384 buys a further 13%, not the ~20% first estimated off an interpolated 4096 point.

## ⚠️ The agent-loop column has been withdrawn — the harness measured unequal work

An earlier version of this note reported an agent loop degrading 37% monotonically with batch size
(1.63 → 2.17 → 2.23 s/turn) and called the trade **established**. That is retracted.

`agentloop.py` sent `max_tokens: 130` with **no `ignore_eos`** and never recorded
`completion_tokens`. `max_tokens` is a **ceiling, not a target**, so each arm was timed on however
many tokens the model happened to emit. A run whose turns stop at 40 tokens beats one whose turns
run to 130 with no difference in speed at all.

The tell arrived on replication: the **no-speculation baseline at batch 4096 gave 1.94 s/turn and
then 1.43 s/turn** — a 36% spread on the same configuration, as large as the entire "trend". Every
agent-loop comparison made that day rests on this harness, so all of them go with it:

- "ngram n=4 beats no-speculation (1.71 vs 1.94)" — **withdrawn**
- "suffix and MTP are worse than baseline (2.13 vs 1.94)" — **withdrawn**
- "batch size trades deep TTFT against agent latency" — **withdrawn**; only the TTFT half survives

This is the same defect this repo already documents under "an all-empty cell is not a comparison":
a measurement that does not assert what it compared. The guard existed in the notes and was not in
the harness. **A guard does not travel to the next script you write** — it has to be in the tool.

**Fixed:** `ignore_eos: True` pins the work at 8 × 130 tokens, per-turn token counts are recorded,
and the summary prints `ms/tok` plus a loud `!! UNEQUAL WORK` flag if the total is not exactly
1,040. Re-running the comparisons on the fixed harness is the open item.

Related: [[agentic-speed-is-ttft-bound]] — still supported by the TTFT column, which is where the
depth cost actually lives, and which this defect does not touch.
