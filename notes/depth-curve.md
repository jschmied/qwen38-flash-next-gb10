# The depth curve: prefill and decode are both flat to 60k

Measured 2026-08-30. One GB10, vLLM `0.1.dev20073+g8e685d198`, FP8-head checkpoint,
`--max-model-len 65536`, `--max-num-seqs 16`, MTP k=2, c=1, 3 requests per depth, 128 output
tokens. Prefill tok/s is derived as `depth / TTFT` so it is directly comparable to the published
curves it is set against below.

| depth | TTFT | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 4,000 | 1.884 s | 2,123 | 37.9 |
| 8,000 | 3.699 s | 2,163 | 40.8 |
| 16,000 | 6.915 s | 2,314 | 41.2 |
| 32,000 | 13.511 s | 2,368 | 41.4 |
| 60,000 | 29.949 s | 2,003 | 41.7 |

## Why this note exists

Because we did not have this curve, and twice quoted a number for it anyway. A prefill figure of
"~320 tok/s at 32k" was carried over from the Qwen3.8-**27B** work and used to conclude that a
competitor's llama.cpp recipe was ahead of us on prefill. Neither the figure nor the conclusion
belonged to this model. **Measuring it took one sweep.**

## What it says

**Prefill is flat, and it is fast.** 2,000–2,370 tok/s across a 15× span of context. The dip at
60k (2,003) is the only departure and is within the run-to-run spread we see elsewhere.

**Decode does not decay with depth.** 37.9 at 4k against 41.7 at 60k. The rise is barely outside
our 6.9% noise floor, so the honest reading is **flat**, not rising — but flat is itself the
result. For a dense-attention model decode falls with context; here it does not, which is what
QSA's top-k selection should produce, since the attention work per decoded token stops growing
once the sparse budget binds. This is the first evidence we have for that on our own hardware.

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
already strong and flat**, and TTFT at realistic agent depths (1.9 s at 4k, 6.9 s at 16k) is not
where the time goes. The ranking moves to the drafter-MoE quantization, which is worth ~3.6 GiB and
reopens three MoE backends.
