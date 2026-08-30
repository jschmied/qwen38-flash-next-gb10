# The depth curve: prefill and decode are both flat to 60k

Measured 2026-08-30. One GB10, vLLM `0.1.dev20073+g8e685d198`, FP8-head checkpoint,
`--max-model-len 65536`, `--max-num-seqs 16`, MTP k=2, c=1, 3 requests per depth, 128 output
tokens. Prefill tok/s is derived as `depth / TTFT` so it is directly comparable to the published
curves it is set against below.

| depth | TTFT (MTP k=2) | prefill | decode | TTFT (MTP off) | prefill | decode | MTP decode gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 1.884 s | 2,123 | 37.9 | 1.902 s | 2,103 | 26.8 | **+41%** |
| 8,000 | 3.699 s | 2,163 | 40.8 | 3.361 s | 2,380 | 26.8 | **+52%** |
| 16,000 | 6.915 s | 2,314 | 41.2 | 7.644 s | 2,093 | 26.8 | **+54%** |
| 32,000 | 13.511 s | 2,368 | 41.4 | 14.981 s | 2,136 | 26.9 | **+54%** |
| 60,000 | 29.949 s | 2,003 | 41.7 | 25.636 s | 2,341 | 27.1 | **+54%** |

## Why this note exists

Because we did not have this curve, and twice quoted a number for it anyway. A prefill figure of
"~320 tok/s at 32k" was carried over from the Qwen3.8-**27B** work and used to conclude that a
competitor's llama.cpp recipe was ahead of us on prefill. Neither the figure nor the conclusion
belonged to this model. **Measuring it took one sweep.**

## What it says

**Decode is depth-independent, and that is the headline.** With speculation off it moves from
**26.8 to 27.1 tok/s across a 15× span of context** — 1.1%, far inside the 6.9% noise floor. For a
dense-attention model decode falls with context because attention work grows with every token
retained. Here it does not move at all. That is exactly what QSA's top-k selection predicts: once
the sparse budget binds, the attention work per decoded token stops growing. This is the first
direct evidence of it on our own hardware, and the MTP-off arm is the clean demonstration because
it has no speculation dynamics on top.

**Prefill is flat and is unaffected by speculation.** 2,003–2,380 tok/s across both arms with no
systematic separation (means: 2,194 with MTP, 2,211 without). Whatever ordering the individual rows
suggest is run-to-run spread, not an effect.

**MTP is a ~1.54× decode win that does not erode with depth.** +41% at 4k, then flat at +52–54%
from 8k out to 60k. It costs nothing at prefill.

### This contradicts a field caution, and the contradiction is worth stating

A llama.cpp recipe reports `draft-mtp` running *slower* than plain autoregressive at 229k on their
QSA kernels, and costing 8.4% / 6.7% of prefill at 16k / 32k. **Neither reproduces here.** Our MTP
advantage is flat to 60k and our prefill is untouched. Different runtime, different kernels, and
their fastest arm is a hand-patched QSA path we do not have — so both results can be true. But the
caution does not transfer to this stack, and we should not have carried it into our own planning
without measuring, which is what this sweep was for.

## Against the field, with the caveats attached

| 32k, c=1 | prefill tok/s | decode tok/s |
|---|---:|---:|
| llama.cpp `UD-IQ4_XS` (DJLougen) | 415 | 24.0 |
| vLLM NVFP4+NVFP4-PLE (spark-arena `e9307821`) | ~1,470 | 17.8 |
| **here** (NVFP4 experts + FP8 dense + FP8 head) | **2,368** | **41.4** |

Both external rows use different quantization, and the llama.cpp row a different harness, so these
are not controlled comparisons. What survives the caveats is the **shape**: spark-arena's prefill is
also flat (1,231–1,600 across 4k–100k) at roughly two-thirds our level, while their *decode* falls
with depth (16.2 → 13.4) where ours does not. The most likely reason for both gaps is the part of
our stack that is not in theirs — FP8 dense projections and an FP8 `lm_head` — since the published
NVFP4 checkpoint leaves the dense projections in BF16 and they are read on every token.

## Consequence: prefill is not our top lever

It was ranked first on the strength of the borrowed number. With the curve measured, **prefill is
already strong, flat, and speculation-neutral**, and TTFT at realistic agent depths (1.9 s at 4k,
6.9 s at 16k) is not where the time goes. The ranking moves to the drafter-MoE quantization, which
is worth ~3.6 GiB and reopens three MoE backends.

**And MTP stays on at every depth.** The one configuration question this sweep might have reopened —
whether to drop speculation for long-context work — is settled in the other direction.
