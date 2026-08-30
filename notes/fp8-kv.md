# FP8 KV on the QSA path: measured, corroborated, and what it is actually worth

Measured 2026-08-30 on one GB10, vLLM `0.1.dev20073+g8e685d198`, FP8-head checkpoint, TP=1,
MTP k=2, PLE CPU offload, `--max-model-len 262144`. KV dtype the only variable between arms.

| | bf16 | fp8_e4m3 | |
|---|---:|---:|---|
| GPU KV cache | 1,077,542 tok | **1,853,358** | **×1.72** |
| max concurrency @262k/req | 4.11× | **7.07×** | ×1.72 |
| needle-in-a-haystack, 5 depths | — | **5/5** | |
| decode, c=1, 4k input (n=6) | 36.28 mean | **37.22 mean** | **+2.6% — no regression** |

## This reverses our own closure from the same morning

Earlier today this was closed as **"unsupported, not unmeasured"** — vLLM's QSA backend declares
`supported_kv_cache_dtypes = ["auto", "bfloat16"]` and enforces it in four places, so `fp8_e4m3` is
rejected at config time.

That was accurate about *stock* vLLM and **wrong as a verdict on feasibility**. The guards were
guarding an **unimplemented read path**, not a hardware limit, and vllm#54426 published a working
17-hunk patch the same day. *"The code refuses it"* and *"it cannot work"* are different claims, and
the four guard citations invited the second reading.

## Corroboration, and why the ratio matters more than the totals

The RFC author measured on their own GB10 and asked for a second machine. Ours:

| | their box | ours |
|---|---:|---:|
| KV pool | 780,638 → 1,399,848 | 1,077,542 → 1,853,358 |
| ratio | ×1.79 | **×1.72** |

Absolute numbers differ ~38% because the checkpoints differ — ours has FP8 dense projections and an
FP8 `lm_head`, leaving more unified memory for KV. **The ratio reproducing while the totals do not
is the stronger result**: the effect scales with whatever KV budget a checkpoint leaves, rather than
matching one machine's figure by coincidence.

## There is no speed gain, and there was never going to be

Three independent lines agree:

1. **Predicted.** sgl#36797 measures fp8_e4m3 at 56.8–58.6 tok/s against bf16's 54–59 on SM121 —
   speed-neutral.
2. **Mechanism.** QSA is *sparse*: with `indexer_budget = 2048` the model attends over a top-k
   selection, not the whole cache, so KV bandwidth is not the decode bottleneck.
3. **Our own depth curve proves it independently** — decode is flat at 26.8 → 27.1 tok/s across a
   15× context increase ([[depth-curve]]). If reading the cache were the constraint, decode would
   fall with depth. It does not.

Halving the precision of something that is not the bottleneck buys nothing, and the added dequant on
the read path plausibly explains the small dip.

**What it buys is admission, not latency.** At 262k tokens per request, concurrent capacity goes
from ~4 to ~7 requests before a queue forms. That is a serving-capacity change.

## The number that was open, and the false pattern in it

At n=2 per arm decode read **−5.6%**, and — the part that made it look real — **both** fp8 runs sat
below **both** bf16 runs. That is a direction rather than scatter, so I reported it upstream as
unresolved rather than dismissing it.

**At n=6 per arm it disappears and the sign flips:**

| | n | mean | median | min–max | sd |
|---|---:|---:|---:|---:|---:|
| bf16 | 6 | 36.28 | 36.60 | 34.0–38.7 | 1.66 |
| fp8_e4m3 | 6 | **37.22** | 36.25 | 35.7–41.2 | 2.10 |

**+2.6%**, ranges overlapping, **0 of 6** fp8 runs below the bf16 range. With sd ≈ 1.7–2.1 on a ~36
tok/s mean, two samples per arm cannot resolve 5% — and "both below both" is exactly the pattern
two samples produce by chance about a quarter of the time. The caveat is withdrawn upstream.

**The rule this reinforces, from our own README:** nothing under ~10% is callable from a single run.
The failure here was subtler than ignoring that — I *did* hedge, but I let an ordering pattern in
four data points carry information it could not carry.

Quality was checked as **retrieval**, not text identity, deliberately: temperature 0 is not
reproducible on this model ([[temp0-nondeterminism]] — five identical requests give five distinct
outputs from ~30 tokens on), so an identity-based check would be meaningless while retrieval is
unaffected. The two findings from today serve each other.
