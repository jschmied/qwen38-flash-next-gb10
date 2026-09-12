POSTED 2026-09-12 (user go "post all"). vllm#55122 reply to @MaCoredroid.
---
Good catch, and it was our own doing: `7cfd04a3` raised the threshold and left the test at the old
boundary. Fixed in `a7188289e` — `seq_len` is now `22015 / 22016 / 22017`.

Verified on hardware rather than just swapping the literals, since the point of the test is which
path runs. Rebuilt this branch's kernel on a GB10 (sm_121) and ran the test body at both
parametrizations:

| `seq_len` set | distinct paths exercised |
| --- | --- |
| 16383 / 16384 / 16385 | **1** — all single-CTA, no transition |
| 22015 / 22016 / 22017 | **2** — 22016 single-CTA, 22017 cooperative |

Both sets reproducible and exact over 4 repeats, so the old widths were passing without testing what
they claimed.

I also took your second point into the docstring: `num_rows=64` can select FilteredTopK on devices
with ≥128 KiB of opt-in shared memory, so there it is correctness coverage rather than transition
coverage. GB10 reports 100 KiB, which is why we never see that branch here.
