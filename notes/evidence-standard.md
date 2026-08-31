# What counts as a finding here

Ten claims were withdrawn on 2026-08-31. Not one was withdrawn because the box misbehaved; every
one was a claim published ahead of its evidence. The causes were four, and they repeat:

| cause | withdrawn claims |
| --- | --- |
| **n=1 read as a result** | "n=6 is 39% worse per turn"; "batch 16384 is a 12% TTFT win"; "n=5 is the deep-context optimum at 45.7"; "k=2 is uniquely anomalous" |
| **a harness that measured the wrong thing** | "ngram beats baseline 1.71 vs 1.94"; "batch size trades TTFT against agent latency" — both from `max_tokens` without `ignore_eos` |
| **one workload generalised** | "`ngram_gpu` is a null" — true on prose, wrong on edit-shaped work, where it is the fastest config |
| **a mechanism argued rather than measured** | the PLE offload shadowing analysis, corrected upstream by a reviewer |

## The rules

1. **No claim from a single arm.** n≥2 to state a direction, n≥3 to state a magnitude. A single arm
   may be recorded as *provisional* and must be labelled so.
2. **Variance is per-configuration, not per-metric.** Most configs here replicate to <1%; `k=4`
   spans 22% and `TTFT@28k at batch 16384` spans 31% while the same metric at batch 4096 spans 3%.
   Never carry an error bar across configurations — this produced two of the four n=1 retractions.
3. **The effect must exceed that config's own spread by a clear margin.** ngram vs no-spec is 23%
   against ~1% spread: sound. A 12% gap against a 22% spread is not a finding.
4. **Every harness asserts what it measured.** `tools/agentloop.py` pins work at 8 × 130 tokens and
   prints `!! UNEQUAL WORK` otherwise. A guard written in a note does not travel to the next script.
5. **Cross-check against an independent instrument.** `ms/tok` must agree with `1000/decode_tps`
   from the separate benchmark; it now does to ~1%. Disagreement means one of them is wrong, and
   that check is what exposed the broken harness.
6. **Interleave controls.** No-spec arms spread across the session gave 43.37 / 43.64 / 43.80
   (1.0%) over five hours, which is how we know the box did not drift and that config differences
   are real. Do this deliberately, not by accident.
7. **State the prediction before the discriminating run.** The ring-capacity hypothesis was written
   down with its two possible outcomes, then refuted by one arm. That is the cheapest possible way
   to be wrong — and it cost one server start where a k=1..8 grid would have cost eight.
8. **Publish only what has replicated.** Notes may hold provisional data, marked. The page and
   upstream get replicated claims only. Four of today's retractions were already public when they
   broke.

## Current findings, graded

**Solid** — replicated, effect ≫ spread, mechanism understood:
- ngram n=4 is the best generic drafter: 33.3 (n=2, 0.4%) vs no-spec 43.6 (n=3, 1.0%).
- MTP k=2 is *worse than no speculation*: 48.6 (n=3, 4.6%) vs 43.6 (n=3). Margin 11% vs 4.6% spread.
- Speculation costs 17-63% of the KV pool (engine config readout, exactly reproducible).
- Prefix-cache hits plateau under MTP while no-spec/ngram accumulate (per-turn counts, every arm).
- Batch 2048 is ~13% worse than 4096 at depth (n=3 each, non-overlapping).
- Batch 16384 buys nothing over 4096 (n=3; means 11.87 vs 11.77) and costs ~4.5% per turn (n=2, 0.2%).
- The ring-widening patch executes and is safe as far as tested: warning logged, NIAH 5/5, n=5 and
  n=6 serve where they could not before. It unlocks n=5..8 **only** — n=9 needs no widening.
- Ring capacity does **not** drive per-turn cost (k=4 shares capacity 8 with k=2, runs 10-17 ms/tok
  faster; refuted in both k=4 arms).

**Provisional** — single arm or unstable, not to be published:
- `ngram_gpu` is the fastest edit-shaped config (26.18, n=1).
- ngram's advantage grows on edit-shaped work, −23% → −33% (n=1 per config).
- MTP n=9 at 37.2 (n=1).
- k=4 at ~35 (n=2 but 22% spread — replication running).

**Open, no mechanism:**
- Why a 2-token draft costs 23 ms/tok above its decode floor while a 4-token draft costs ~5.
  Acceptance rate is the obvious candidate and is **not instrumented** — we measure wall-clock only.
