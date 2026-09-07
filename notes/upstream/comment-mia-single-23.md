DRAFT — user go given ("post #23 and #19 comments"). MiaAI single-Spark #23 (2026-09-07).

Both halves of this match what we measured on a GB10, and we think the throughput half is an expected
result rather than a shortfall.

**FP8 KV buys pool, not speed — and there was never a mechanism for it to buy speed here.** Two arms,
KV dtype the only variable, `--max-model-len 262144`, TP=1:

| | bf16 | fp8_e4m3 | |
|---|---|---|---|
| GPU KV cache | 1,077,542 tok | 1,853,358 | **×1.72** |
| max concurrency @262k/req | 4.11× | 7.07× | ×1.72 |
| needle-in-a-haystack, 5 depths | — | 5/5 | |
| decode, c=1, 4k input (n=6) | 36.28 mean | 37.22 mean | +2.6 % |

QSA is **sparse**: with `indexer_budget` binding, the model attends over a top-k selection rather than
the whole cache, so KV bandwidth is not the decode bottleneck and halving its precision cannot help.
Our depth sweep shows the same thing independently — with speculation off, decode moves 26.8 → 27.1 tok/s
across a 15× span of context (4k to 60k). Decode that is flat in context is not reading the cache.

**A caution on the sample size, because we fell into it.** At n=2 per arm we measured fp8 −5.6 %, and —
the part that made it look real — *both* fp8 runs sat below *both* bf16 runs. A direction, not scatter.
At n=6 the sign flipped and the ranges overlap. With a standard deviation near 1.7–2.1 on a ~36 tok/s
mean, two samples cannot resolve five percent, and "both below both" happens by chance about a quarter
of the time. Worth n≥6 before concluding fp8 is slower.

Corroboration for the ratio: a second GB10 reports ×1.79 against our ×1.72, while the absolute totals
differ by ~38 % because the checkpoints differ. The ratio reproducing while the totals do not is the
stronger result — it scales with whatever KV budget a checkpoint leaves.

**On the `NVRM ... NV_ERR_NO_MEMORY` events.** We see the same signature and have not been able to make
it mean anything: **268 of those lines in a single burst** at one server transition, with **no process
killed and zero `oom-kill` entries in the kernel log**, and every subsequent run started and completed
normally. Chris's follow-up that it also reproduces on BF16 startup fits — on our box it tracks the
allocator probing during startup/teardown rather than the KV dtype. Two discriminators before treating it
as a stop condition: grep the kernel log for `oom-kill` / `Killed process` (a real OOM names a victim),
and check whether host `MemAvailable` actually dipped. If neither fires, it is the driver reporting a
failed speculative allocation that it then retries.

*AI assistance was used in preparing this comment; the measurements are ours.*
