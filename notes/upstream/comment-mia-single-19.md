DRAFT — user go given ("post #23 and #19 comments"). MiaAI single-Spark #19 (2026-09-07).

Agreed that `MAX_NUM_SEQS=4` is a benchmarking trap, but the 33 → 267 tok/s figure being cited for it is,
we think, measuring a different thing — and the curve has a knee you are on the wrong side of.

**Raising the cap above ~16 is null on this hardware.** SEQS 16 vs 64, nothing else changed:

| | SEQS 16 | SEQS 64 |
|---|---|---|
| c=1 decode tok/s | 36.45 ± 1.04 | 36.83 |
| c=16 aggregate tok/s | 99.1 / 101.5 / 100.1 | 97.2 / 101.4 / 100.7 |
| c=32 aggregate tok/s | not reachable | 110.1, TTFT 30.7 s |

Both moves sit inside our 6.9 % decode noise floor. A separate recipe reports 96–109 tok/s aggregate at
`SEQS=16` on the same hardware with a different checkpoint and speculation setting — two independent
configurations landing on the same number is much better evidence for a ceiling than either alone. And
c=32 buys ~10 % aggregate for 4× the TTFT, which is the wrong trade for agent work.

**The 267 tok/s number is a different configuration.** In our hands that figure came from the *baseline*
checkpoint — whose c=1 row is 17.1 tok/s — with speculation off and short prompts. It is not comparable to
a ~100 measured on a tuned checkpoint at 4,000-token inputs, and reading the two side by side manufactures
a regression that does not exist. Likewise the published 1.2–2.7× "SEQS win" was measured by raising a
baseline of **2** slots, not 16: the same curve, below its knee. So the fix here is real (4 is too low) but
the expected payoff is "reach the knee", not "2.7×".

**One caveat on our own ceiling, prompted by your README.** Our runs pinned `cudagraph_capture_sizes` to
`[1, 2, 4, 8]`. With MTP n=3 each sequence contributes 1+3 rows, so the decode batch is 4×S — meaning
S=4 → 16 and S=8 → 32 were **not** captured widths for us, and those steps may have run eager. Your finding
that a full decode graph at every verify width (4 through 32) is what makes the 8-stream column reachable
is exactly the thing that would explain it. We are re-testing before defending "~100 tok/s is a bandwidth
ceiling" any further; treat our c=16 numbers as provisional until then.

*AI assistance was used in preparing this comment; the measurements are ours.*
