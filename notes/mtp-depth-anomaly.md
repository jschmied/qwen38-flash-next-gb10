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
