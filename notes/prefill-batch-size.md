# Prefill batch size: the default was already right

Measured 2026-08-31, one server start per arm, no speculation, `fp8head`,
`max-model-len 32768`, c=1. TTFT cells are medians of 6 requests; `ms/tok` is the fixed-work
agent loop (`tools/agentloop.py`).

| batch | TTFT @28k (samples) | mean | ms/tok |
| ---: | --- | ---: | --- |
| 2048 | 13.04 / 13.08 / 13.74 | 13.3 | 43.46 / 43.92 |
| **4096 (prod default)** | 11.62 / 11.73 / 11.96 | **11.77** | 43.37 / 43.64 / 43.80 |
| 8192 | 11.78 | 11.78 | — |
| 16384 | 10.41 / 13.64 / 11.57 | 11.87 | **45.52 / 45.41** |

**Conclusion: keep 4096.** Going below it costs ~13% deep TTFT. Going above it buys nothing and
costs ~4.5% per agent turn. Decode is flat at 26.7-27.3 everywhere.

At 4k, batch size does nothing at all (1.60-2.13 across every setting) — a 4000-token prompt is one
chunk at any size ≥ 4096.

## ⚠️ An earlier version of this note recommended 16384. That was wrong.

The first 16384 arm measured **10.41 s** at 28k, which with tight 4096 and 2048 arms either side
produced a clean monotonic trend. It was called *established* and *"the one result today I'd act on
unreserved"*, and shipped to the published page.

Replication broke it: the next two arms gave **13.64** and **11.57**. Batch 16384's deep TTFT spans
**31%** where 4096 spans 3% — the 10.41 was the low end of a noisy distribution, not a trend point.
Engine config was byte-identical across those arms (KV pool 1,011,875, block 800, concurrency
30.88×), so this is not the vllm#54122 compile-cache effect; the configuration is simply noisier at
that batch size.

**The reasoning error is worth more than the result.** Three tight arms each at 2048 and 4096
convinced me the *metric* was stable, so I trusted single points at 8192 and 16384. Stability at one
configuration does not transfer to another — noise is a property of the (metric, configuration)
pair, not of the metric. Every setting needs its own replication before it joins a trend.

The `ms/tok` column is the counter-example that makes the point: at 16384 it reproduces to 0.2%
(45.52 / 45.41) while TTFT at the same setting spans 31%. Same arms, same server, two metrics with
completely different stability.

Related: [[which-drafter-for-agent-work]], [[speculation-costs-kv-pool]].
