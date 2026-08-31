# `FULL_DECODE_ONLY` is a free win: −2% latency and +17% KV pool

Measured 2026-09-01 on an idle box, no speculation, `fp8head`, batch 4096, `max-model-len 32768`,
c=1. **Controls at both ends** — that is what makes the middle two arms readable.

| `cudagraph_mode` | ms/tok | decode 4k / 28k | GPU KV cache | max concurrency @32k | init |
| --- | ---: | ---: | ---: | ---: | ---: |
| PIECEWISE (production) | 43.57 / **43.78** | 27.0 / 26.8 | 993,525 | 30.32× | 36 s |
| **FULL_DECODE_ONLY** | **42.78** | 27.6 / **28.0** | **1,162,608** | **35.48×** | 142 s |
| FULL | 42.55 | 27.6 / 27.3 | 953,548 | 29.10× | 130 s |

The two PIECEWISE controls agree to **0.5%** (43.57, 43.78), so nothing drifted across the run and
the differences are attributable to the mode.

**Ship `FULL_DECODE_ONLY`.** ~2% faster per turn, ~4% faster deep decode, and **17% more KV cache**,
which lifts max concurrency from 30× to 35× at 32k. The only cost is a 4× slower start (36 s → 142 s),
irrelevant for a long-running server and merely annoying for benchmark iteration.

**`FULL` is strictly worse than `FULL_DECODE_ONLY`**: same latency, but capturing prefill graphs
costs 18% of the pool relative to decode-only capture (953,548 vs 1,162,608) — *below* the
PIECEWISE baseline — and buys nothing for it.

## Why the latency gain is small, and why that is the right result

Hyper-connections are ~25% of decode GPU time, and graph capture is the one lever that removes
**launches** rather than bytes — which is why it was chosen after three quantization attempts
measured null. But they sit at **~78% of roofline**, so launch overhead was never the bulk of their
cost. A ~2% gain is what that diagnosis predicts. The result confirms the model rather than
overturning it, and it closes the "maybe there is 25% sitting there" question.

## The memory result was the surprise, and it went against the prediction

I expected full capture to **cost** KV memory and said so before the run. It **gains 17%**.
Mechanism, in hindsight: `PIECEWISE` must keep intermediates live across each piecewise boundary,
while a fully captured decode graph reuses one fixed allocation. Capturing *more* can use *less*.

⚠️ **n=1 per mode.** The controls are bracketed and tight, but the modes themselves want replication
before this goes to production. Two arms each would settle it.

Related: [[speculation-costs-kv-pool]] (the other lever that moves the pool), [[evidence-standard]].
