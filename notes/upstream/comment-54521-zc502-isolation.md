DRAFT — needs the user's go. GitHub vllm-project/vllm issue #54521, reply to ZC502 (2026-09-08).

@ZC502 The client collector works on sm_121 — thank you for adding it. It has since carried real
load here: it was the measurement instrument for a five-arm isolation run, and it did not need a
single change.

**Validation**, GB10 / sm_121 / TP=1, Qwen3.8-Flash-Next NVFP4, `--execution-mode sequential
--repeats 8 --prompt-logprobs 5 --max-tokens 64 --temperature 0`: 12/12 runs completed, output
schema stable, and `analyze.py`'s `positions_with_top1_disagreement` reproduced our own
divergence counts on the same traces. One usability note that cost us a run: `analyze.py` takes
`reference [candidate]` as **files** and `--out` as a **directory** — passing a directory as the
reference fails late and unhelpfully. A line in the README would save the next person the round trip.

**What we used it for, since it bears on this issue directly.** The thread now has five separate
defects from five reporters, and nobody had said what fixing *one* of them buys. On our box the answer
is: nothing observable. One fix active at a time against a no-fix control, per-arm compile and autotune
cache roots purged between arms, 2,504-token prose prompt, 8 sequential repeats — the control shows 333
disagreeing positions, each single fix leaves **280–334**, and all four together leave **0**. The table is
in the comment I posted a moment ago for @mmastrac, so I will not repeat it here.

Two consequences for this issue, offered as method rather than as a claim about anyone else's hardware.
**A single-fix A/B on a machine with more than one of these defects will read as null**, which is a good
way to discard a correct fix, and may be what several reporters here are seeing. And if the collector grew
a way to record *which* fixes an arm carried in its metadata, that distinction would survive into the
analysis instead of living in someone's notes.

Raw data and the runner's validity gate (the control must misbehave, the all-fixed arm must not):
[`notes/data/isolate5.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/bc88bc377a64fca7518eba80d793a7e25050b425/notes/data/isolate5.txt),
write-up as finding 184 in
[`notes/determinism-investigation.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/bc88bc377a64fca7518eba80d793a7e25050b425/notes/determinism-investigation.md).

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
