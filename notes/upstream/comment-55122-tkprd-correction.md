DRAFT — user go given ("post correction det-146"). vLLM PR #55122 (2026-09-07).

Correcting my own numbers above: **the `top_k_per_row_decode` comparison I posted is stale, in my favour to
retract.** I measured it against the kernel as it stood then; the two commits since (merge instead of sort,
one-warp bin scan, skipped radix passes, and the dead-sort cleanup) made this kernel 2–4× faster on the
shapes that were worst, so "0.15–0.92× of our det kernel" no longer holds.

Re-measured on the same box, same harness, against the current head:

| shape (rows / n / k) | tkprd / this PR, **as I posted it** | tkprd / this PR, **now** |
| --- | --- | --- |
| 1 / 16,384 / 2048 | 0.50 | **1.18 — this PR is faster** |
| 1 / 16,384 / 512 | 0.74 | **1.14 — this PR is faster** |
| 8 / 16,384 / 2048 | — | **1.13 — this PR is faster** |
| 1 / 32,768 / 2048 | 0.66 | **1.08 — this PR is faster** |
| 1 / 32,768 / 512 | 0.92 | 1.00 — parity |
| 64 / 16,384 / 2048 | 0.39 | 0.98 — parity |
| 1 / 8,192 / 2048 | 0.27 | 0.80 |
| 64 / 8,192 / 2048 | 0.18 | 0.53 |
| 64 / 32,768 / 2048 | 0.28 | 0.43 |

**Range 0.42–1.18×, was 0.15–0.92×.** The split is structural rather than noise: `top_k_per_row_decode` is
faster on **many rows × short rows**, this kernel is faster on **few rows × long rows**. The QSA decode
shape is the latter — MTP with n=3 at a single stream issues 4 query rows — so in the regime this dispatch
actually runs, the two are at parity or this PR is ahead.

Nothing about correctness moved: `top_k_per_row_decode` is still **not deterministic on any of the 56
shapes** I tested, and its selected set still differs from the exact reference on every tie-heavy shape.

The follow-up I offered — giving that kernel the same index-ordered emission — is therefore less attractive
than when I offered it, since half the argument for it was that it is also faster. I am still happy to do it
if someone who owns that kernel wants it, but I would not now claim it dominates this one on speed.

*AI assistance was used in preparing this comment; the measurements are ours.*
