# Speculation's real price on this model is the KV pool, not TTFT

Measured 2026-08-31 from the engine's own startup lines, `fp8head`, `max-model-len 32768`,
`gpu-memory-utilization 0.90`, batch 4096 unless noted.

| arm | attention block size | GPU KV cache | vs no-spec | max concurrency @32k |
| --- | ---: | ---: | ---: | ---: |
| no speculation | 800 | 985,006 tok | — | 30.06× |
| no speculation, batch 16384 | 800 | 1,011,875 tok | +2.7% | 30.88× |
| ngram n=4 | 848 | 817,664 tok | **−17%** | — |
| MTP k=2 | 1600 | 627,765 tok | **−36%** | 19.16× |
| MTP n=6 | 1632 | 430,231 tok | **−56%** | — |

The mechanism is in the log: *"Setting attention block size to N tokens to ensure that attention
page size is >= mamba page size."* Speculation **doubles** the attention block size for MTP
(800 → 1600), and a larger page is a coarser allocation unit, so usable KV falls hard. Enabling
MTP k=2 costs **more than a third of the KV pool**; n=6 costs more than half.

This reframes the drafter question. Everything measured earlier was single-stream at short context,
where the pool is irrelevant — so speculation looked nearly free apart from the one-block prefix
back-off. It is not free at all if the box serves concurrency or long context:

- **ngram n=4 is the cheap drafter**: 848 vs 800 block, −17% pool, and +15-18% decode. It never
  triggers the `use_eagle` back-off either ([[mtp-vs-prefix-cache]]).
- **MTP k=2 costs −36% pool** for its decode gain. Whether that trades well depends entirely on
  whether concurrency or context depth is the binding constraint, and single-stream benchmarks
  cannot see it.
- **MTP n=6 costs −56%** — combined with it being slower per agent turn, the deep band looks worse
  the more ways it is measured.

## It also confounds every cross-drafter comparison made today

The arms do **not** share an allocation unit — 800, 848, 1600, 1632. Block size is the
prefix-cache hash unit, so a `ms/tok` comparison across drafters is partly a comparison of cache
geometry. Prompts landing near a boundary favour one arm arbitrarily. Fixes: hold prompt lengths
well off the boundary, and report `prefix_cache_hits_total` deltas per arm rather than trusting
wall-clock alone.

**Method note:** this cost was sitting in the startup log of every arm run today and went unread
for eight hours while the same arms were analysed for tok/s. Read what the engine reports about the
configuration, not only what the benchmark reports about the run.
