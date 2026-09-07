DRAFT — needs the user's go. GitHub vllm-project/vllm issue #53670 (2026-09-07).

One correction to my own numbers, since both of you have read them as "acceptance flat".

They were not flat — acceptance was consistently **higher without** the drop: 56.1 / 53.3 / 53.8 %
with it against 59.5 / 57.5 / 60.0 % without, three interleaved starts each, ranges not overlapping.
So the blanket drop did not merely fail to buy acceptance on this layout, it cost about 4–6 pp of it,
on top of the −26 % warm-turn latency. @Suppressor72, that makes your point stronger than you put it.

Worth stating for calibration: this is an always-drafting config (MTP n=3, K>0 on every admission),
so the K=0-consumer proposal would not change our path at all. We are a data point for it, not a
beneficiary of it.

Agreed on keeping #54360 separate.
