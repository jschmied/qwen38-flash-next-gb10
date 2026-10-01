DRAFT — needs the user's go. ashhart/TensorFold #136 reply to mrpmorris (2026-10-01).

@mrpmorris fair point: the sweep tested thresholds, not your controller. Here's a replay of it on our logged drafts. The
log is the same model on vLLM on a GB10: 567 code and 950 prose verify steps of 7 drafts each, with the drafter's
probability at every position. It's priced with measured verify costs for each draft count. The controller calibrates
the drafter's probability to real acceptance on half the steps and is scored on the other half, both ways. After each
draft it picks the depth with the highest expected tokens per ms.

| vs fixed 5 drafts | code | prose |
|---|---|---|
| confidence gate 0.7, up to 7 drafts | +10.0 % | +12.6 % |
| expected-throughput controller | +8.0 % | +11.5 % |
| oracle: draft exactly the accepted chain | +25.4 % | +34.7 % |

The controller drafts a little more than the gate (5.5 against 4.9 drafts a step on code) and ends 1–2 % behind it.
Calibrating by position as well doesn't change that. So there's real headroom over the gate, 14–20 % with perfect
knowledge, but on this log the drafter's probabilities don't predict acceptance well enough to reach it.

Caveats: these are vLLM's costs, not TensorFold's; steps are treated as independent; and it's one log. A live A/B on
TensorFold would need the controller inside the draft loop. Script and data:
<SCRIPT_LINK>.

Written with AI assistance (Claude Code).
