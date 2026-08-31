# MTP k=2 is anomalously slow, and it is not the ring capacity

Fixed-work agent loop (`tools/agentloop.py`, 8 × 130 tokens, `ms/tok`), `fp8head`, batch 4096, c=1.

| depth | QSA ring capacity | ms/tok | arms |
| --- | ---: | ---: | ---: |
| no speculation | — | 43.6 | 3 |
| **k=2** | 8 | **48.6** | 3 |
| k=4 | 8 | **31.9** | 1 |
| n=5 | 16 (widened) | 31.9 | 2 |
| n=9 | 16 | 37.2 | 1 |

**k=2 is the only configuration on this box slower than no speculation at all**, and it is not a
depth trend: k=4 and n=5 sit together at ~32, n=9 degrades to 37, and k=2 is off on its own at 48.6.

## The capacity hypothesis, stated in advance and refuted

Capacity is `compress_ratio * cdiv(compress_ratio + n, compress_ratio)`, giving 8 for n=1..4 and 16
for n=5..12. The three points we had (k=2 bad, n=5 good, n=9 good) lined up exactly with capacity 8
versus 16, and the prediction was explicit: **k=4, the deepest capacity-8 setting, should land near
48 if capacity drives the cost, near 32 if depth does.**

It measured **31.93**. Capacity does not drive it. The correlation was three points arranged so that
capacity and depth could not be told apart, and one arm separated them.

Worth keeping because the alternative was expensive: a full k=1..8 grid is eight arms (~100 min);
the discriminating arm was one. Pick the cell that splits the hypotheses, not the cells that fill
the table.

## What the anomaly is not

- **Not the prefix-cache back-off.** Per-turn hit counts plateau identically for every MTP depth
  (k=2 → 4800, n=5 → 4848, n=9 → 4992) while no-spec and ngram grow to ~8000. All MTP depths pay the
  `use_eagle` block equally; only k=2 is slow ([[which-drafter-for-agent-work]]).
- **Not decode.** k=2 has among the best decode measured (37-42 tok/s at 4k, ~41 at 28k). k=4's is
  higher still (42.8) with a third of the per-turn cost.
- **Not the KV pool.** k=2 costs 36%, n=5 costs 52%, n=9 costs 63% — the cheapest of the three on
  that axis is the slow one ([[speculation-costs-kv-pool]]).

## Open

`k=3` is running: near 32 means k=2 alone is broken; near 48 means shallow depths are collectively
broken and the threshold sits between 3 and 4. Either answer is sharper than the question we
started with, and neither is explained by anything measured so far.

The field is silent here — nobody else reports agent-loop latency on this model, and the one public
ring-width A/B (hashd1ve, SGLang) confounds width with depth, comparing ring-4-at-depth-4 against
ring-8-at-depth-8.


## Quiet-box re-measurement (2026-08-31, after the contamination was found)

Five arms between 20:18 and 21:36 were invalidated by research subagents running on the same box
([[read-only-is-not-load-free]]). Re-run quiet:

| depth | ring capacity | quiet ms/tok | note |
| --- | ---: | ---: | --- |
| k=2 | 8 | 48.6 (n=3) | all three pre-date the contamination |
| k=3 | 8 | 32.48 | 21:05, between agent windows |
| k=4 | 8 | 31.93, **31.47** | both quiet; the 38.96/57.61 arms were contaminated |
| n=5 | **16** (widened) | 31.81, 31.97 | |
| **n=6** | **12** | **72.32** | quiet — **confirms the contaminated 71.85** |
| n=9 | 16 | 37.23 | |

⚠️ **Correction:** after finding the contamination I guessed n=6's results were "probably artifacts
too". They are not. A quiet arm returned 72.32 against the contaminated 71.85 — **n=6 is genuinely
2.3× slower than its immediate neighbours.** Contamination inflated k=4; it did not create n=6.

## The capacity hypothesis, in the only form that still fits

Refuted twice already (k=4 and k=3 both share capacity 8 with k=2 and are fast). But n=6 is the
**only configuration on a capacity-12 ring**, and it is the slowest thing measured:

- capacity **8** → 48.6 (k=2), 32.5 (k=3), 31.7 (k=4)
- capacity **12** → **72.3** (n=6), alone
- capacity **16** → 31.9 (n=5), 37.2 (n=9)

If capacity 12 is pathological, it also explains why our widening patch makes n=5 the *best* config
(12 → 16) while n=6, whose block size happens to be divisible by 12, keeps the bad ring natively and
is the worst. That would make the patch a **performance** fix at the depths where it fires.

**k=2 remains a separate anomaly** — capacity 8, same as the fast k=3 and k=4, yet 48.6 across three
tight arms. One hypothesis does not cover both.

**Discriminating arm, prediction fixed in advance:** n=7 has span 11 → capacity 12. If its block
size is divisible by 12 the ring stays at 12 and this predicts **~70**; if the widening fires it
runs 16 and this predicts **~32**. Read the widening line from the log *before* the `ms/tok`.
