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

+9.9% at 4k and +12.4% at 16k at n=1 — but **this did not survive confirmation and is withdrawn**.

**Refuted at n=3** (8k input, same box, same day): 4096 mean **2,212** tok/s against 8192's
**2,005**, with a within-arm spread of **1,633–2,367** — a 45% range in a single configuration. The
setting stays at 4096.

**The transferable finding is the variance itself.** Our 6.9% noise floor is a *decode* figure;
prefill at 8k swings **±20%** run to run. Every single-run prefill number we have quoted carries
that, and `depth-curve.md` is smooth because its points are three-request medians rather than
because prefill is stable. Hedging this as *adopt after one confirmation* is the only reason it
never reached the shipped config.

### An unplanned control worth noting

Steady-state cache hits were **identical across three independent server starts** (1,600 / 4,800 /
12,800 / 27,200 / 56,000 at the five lengths, for all three batch sizes). Only the transient varies.
That means the cache measurement has essentially no run-to-run variance, so the k=0 deficit above is
a real effect and not scatter — which is what makes a one-run-per-cell design defensible *for this
particular quantity*, and not for the decode numbers beside it.

Decode also reproduced the QSA signature to three significant figures on a different day and batch
size: k=0 reads **26.9 / 26.9 / 26.9** across 4k/16k/32k.

## The threshold, confirmed by prediction (2026-08-31, 06:02–06:40)

The earlier agent-loop runs only ever sampled *one side* of the break-even, and a first attempt to
test the other side was invalid: raising `max_tokens` from 130 to 400 changed nothing, because the
prompt says *"briefly"* and the model generated **45 tokens** either way. **`max_tokens` is a
ceiling, not a target.** Redone with `ignore_eos` forcing the length, and every turn recording its
actual `completion_tokens`:

| forced turn length | MTP k=2 | MTP off | |
|---:|---:|---:|---|
| **30 tokens** | 19.4 s (2.42/turn) | **16.0 s** (2.00/turn) | no-MTP **+17.5%** |
| **400 tokens** | **92.6 s** (11.57/turn) | 122.2 s (15.27/turn) | MTP **+24.2%** |

**The reversal is real**, and solving `advantage(t) = a·t − b` from the two points recovers the model
that predicted it:

| | predicted from the code | measured |
|---|---:|---:|
| decode saving `a` | 0.01331 s/tok (`1/26.9 − 1/41.9`) | **0.01114** |
| fixed block cost `b` | 0.667 s (`1600 / 2400 tok/s`) | **0.754** |
| break-even | ~50 tokens | **68 tokens** |

Both coefficients within ~15% of values derived *before* the measurement, from
`scheduler.py:397` and the depth curve. That is what separates this from the three claims withdrawn
the same night — those were patterns found in data and rationalised afterwards; this one predicted a
regime it had not seen.

## The rule

> **Run MTP k=2 when turns exceed ~68 output tokens. Disable it below that.**

Not a compromise setting — a threshold, with both sides measured and a mechanism that explains why
it sits where it does. Practically: ordinary generation is far above it, and terse tool-calling
loops are below it. Never k=1 (dominated). The shipped default stays **MTP k=2**, which is correct
for everything except short-turn agent work.

## The 5..8 band is now reachable — and n=6 is worse than no speculation (2026-08-31)

Two separate findings, and the second undercuts the reason for wanting the first.

**The depth limit was a misread constraint.** `k=5` fails with `QSA ring capacity 12 must divide
the attention block size 848`, and we had recorded that as a ceiling. It is a hole:
`capacity = compress_ratio * cdiv(compress_ratio + n, compress_ratio)`, so n=0..4 and n=9..12 are
legal and only 5..8 are not (848 = 2^4 * 53 has no factor 3). The requirement is also **one-sided** —
the code's own comment says a ring narrower than `span` lets a rejected draft row overwrite a
committed key, so wider is merely slack. Widening 12 -> 16 instead of asserting unlocks the band;
see `patches/models_qwen3_8_flash_next_common_qsa_cache.py.patch`.

**This is not a GB10 quirk.** I first read it as one, because our 848 comes from the hybrid-group
LCM and looks exotic. It isn't: `n = 5..8` all give capacity 12, which needs a factor of 3, and no
power-of-two block size has one — so the band is unreachable for **every user of this model on
every GPU**, and the assert is still on `main`.

| block_size | blocked n |
| --- | --- |
| 16 / 32 / 64 / 128 / 256 / 512 / 1024 / 2048 | 5, 6, 7, 8, 13, 14, 15, 16 |
| 848 (ours) | 5, 6, 7, 8, 13, 14, 15, 16 |
| 1600 | 5, 6, 7, 8 |

Reported upstream as [vllm#54552](https://github.com/vllm-project/vllm/issues/54552), including
both caveats against our own case: the unobserved warning, and the fact that the band we unlocked
is a regression *for our workload* while a SGLang DGX Spark report measures accepted length
3.95 -> 7.12 from a wider draft window on code completion. Reachable and measurable is the claim;
better is not.

Verified rather than assumed: pre-patch, capacity 12 hard-failed for **two different drafters**
(ngram n=5 at 12:01, suffix n=5 at 12:22), so `block_size 848` is stable across methods. Post-patch,
`SpeculativeConfig(method='mtp', num_spec_tokens=6)` serves. ⚠️ The widening `warning_once` never
appeared in the log — the branch either ran silently or was deduplicated. The assert provably fired
before and provably does not now, but that line should be visible before this goes upstream.

**And then n=6 loses to no speculation at all on agent work.** Same harness, same checkpoint, same
batch 4096 / maxlen 32768:

| | decode | TTFT | agent loop, 8 turns |
| --- | --- | --- | --- |
| no speculation | 27.0 tok/s | 1.60 s | **1.94 s/turn** |
| MTP n=6 | 35.2 tok/s (+30%) | 2.63 s (+65%) | 2.70 s/turn (**+39% worse**) |

+39% is far outside the 6.9% decode noise floor. The decode column and the agent column point in
opposite directions, which is the whole thesis of [[agentic-speed-is-ttft-bound]] showing up in one
arm: deeper speculation buys decode and pays in TTFT, and a ~130-token turn cannot amortise the
prefix block it costs.

**So the 9..12 band is not the speed lever it looked like this morning.** It is a decode-benchmark
lever and an agent-work regression. Still worth one arm if long single-shot generation ever matters
here — break-even is ~68 output tokens — but it should not be swept on the strength of the
formula alone. Measuring n=6 cost one server start and closed the whole band for agent use.
