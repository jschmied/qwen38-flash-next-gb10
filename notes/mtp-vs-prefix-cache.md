# MTP costs exactly one cache block, and on agent work that outweighs its decode gain

Measured 2026-08-31 (overnight), one GB10, FP8-head checkpoint, `--max-model-len 32768`.

## The mechanism, from the scheduler

`v1/core/sched/scheduler.py:395-397`:

```python
last_cache_position = request.num_tokens - request.num_tokens % block_size
if self.use_eagle:
    last_cache_position = max(last_cache_position - block_size, 0)
```

with the comment *"With Eagle, FullAttn prunes the last matching block, so back off one block to
avoid a Mamba cache miss."* MTP sits inside `EagleModelTypes`, so `use_eagle` is **true** for us and
the cacheable position backs off **one full block — 1,600 tokens here**.

**It is a fixed cost, not a proportional one.** Our earlier reading of "+1,600 with MTP vs +3,200
without" as a *halving* was an artifact of the test prompt being ~2 blocks long. At 32k it would be
~5%; at 2k it is everything.

## The agent-loop A/B

Eight turns of a growing conversation on a ~8k-token shared prefix, 130-token turns — the shape
agent work actually has, and the shape **none of our published benchmarks use**, because they all
send unique prompts.

| | MTP k=2 | MTP off |
|---|---:|---:|
| total, 8 turns | 18.2 s | **16.4 s** |
| per turn | 2.28 s | **2.04 s** |
| steady-state hits/turn | +4,800 (3 blocks) | **+7,200 (4.5 blocks)** |
| steady-state turn latency | 1.81–2.07 s | **1.11–1.60 s** |

**MTP off is ~10% faster**, despite MTP being worth +54% decode on the depth curve. With short turns
the prefill dominates, the extra 1.5 cached blocks win, and the decode advantage cannot pay the
block back.

One nuance that cuts the other way: **MTP reaches steady state a turn earlier** — full hits from
turn 2, where no-MTP needs turn 3. MTP wins the first exchange and loses every one after.

## And the caching quirk does not apply to agent work

The documented *"first repetition never hits, only the second"* behaviour is about **identical**
prompts. An agent loop sends a **growing** conversation sharing a prefix, and hits land on **turn 2**
— TTFT 4.45 s → ~1.9 s immediately. The quirk is real but does not describe the workload we care
about.

## There is no config that avoids the trade

The back-off is `if self.use_eagle:` — unconditional and one full block **regardless of `k`**. So
MTP k=1 cannot buy back a fraction of it. The trade is structural, and the decision is by workload:

- **short agent turns → MTP off**
- **long generations → MTP on**

consistent with our standing TTFT/decode break-even near ~2,000 output tokens.

## Status: not acted on yet

n=1 per arm, and 1.8 s over 18 s is close enough to the noise floor to demand repetition. Rounds 2
and 3 are queued, alternating arms. **The shipped config still runs MTP k=2** and will keep doing so
until this survives n=3.

## Overnight program: the cost is fixed, and k=1 is dominated

Five arms, each measuring a cache-free depth curve (unique prompts) and cache behaviour across five
prompt lengths (four identical requests each).

### The MTP cache cost is a fixed ~1,600 tokens — one block — at every length

| prompt | k=2 | k=1 | k=0 | deficit vs k=0 | as % of k=0 |
|---:|---:|---:|---:|---:|---:|
| ~2k | 1,600 | 1,584 | 3,200 | 1,600 | **50.0%** |
| ~4k | 4,800 | 4,752 | 7,200 | 2,400 | 33.3% |
| ~8k | 12,800 | 12,672 | 14,400 | 1,600 | 11.1% |
| ~16k | 27,200 | 26,928 | 28,800 | 1,600 | **5.6%** |
| ~32k | 56,000 | 57,024 | 58,400 | 2,400 | **4.1%** |

Absolute cost flat, relative cost collapsing with length — exactly what `scheduler.py:397` predicts.
The occasional 2,400 is block alignment of that particular prompt, not a different effect.

**So above ~8k there is barely a trade at all:** MTP buys +56% decode for ~5% of the cache. Below
~4k it is genuinely expensive.

### k=1 is strictly dominated — delete it from the option space

| | cache cost | decode vs k=0 |
|---|---:|---:|
| k=2 | 1,600 | **+56%** |
| k=1 | 1,584 | +31% |

The back-off is `if self.use_eagle:` — unconditional on `k` — so k=1 pays the **same block** (the
16-token difference is prompt alignment) for **half the decode gain**. There is never a reason to
run k=1 on this model.

### Prefill: `--max-num-batched-tokens` 8192 beats our shipped 4096

| depth | 4096 (shipped) | **8192** | 16384 |
|---:|---:|---:|---:|
| 4k | 2,189 | **2,405** | 2,278 |
| 16k | 2,146 | **2,413** | 2,153 |
| 32k | **2,223** | 2,124 | 2,094 |

+9.9% at 4k and +12.4% at 16k — the band agent work lives in — while 16384 is no better than 4096
anywhere. Gains sit just above the 6.9% noise floor at n=1, so this is *adopt after one confirmation*
rather than settled.

### An unplanned control worth noting

Steady-state cache hits were **identical across three independent server starts** (1,600 / 4,800 /
12,800 / 27,200 / 56,000 at the five lengths, for all three batch sizes). Only the transient varies.
That means the cache measurement has essentially no run-to-run variance, so the k=0 deficit above is
a real effect and not scatter — which is what makes a one-run-per-cell design defensible *for this
particular quantity*, and not for the decode numbers beside it.

Decode also reproduced the QSA signature to three significant figures on a different day and batch
size: k=0 reads **26.9 / 26.9 / 26.9** across 4k/16k/32k.
