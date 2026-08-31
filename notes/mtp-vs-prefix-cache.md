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
