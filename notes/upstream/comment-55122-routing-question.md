DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122 (2026-09-07).

## A question about the `rows > 32` routing, which may be where the filtered-path cost actually lives

Following up on my own numbers: the worst cells are all on `FilteredTopKRaggedTransform`, and I want to
check an assumption before optimising inside it.

`FilteredTopKUnifiedKernel` launches `grid(num_rows)` — **one CTA per row**. At 64 rows on a 132-SM
H100 that leaves more than half the machine idle. That is true of the current kernel too, so it is not
something this PR introduced, but it does mean the dispatch is sending the widest rows to the launch
geometry with the least parallelism. My measurements are consistent with it: going 64 → 256 rows at
n=65,536, the pre-PR kernel scales 28.7 → 68.4 µs and this branch 77.2 → 159 µs — both clearly
sublinear for 4× the work, i.e. there is capacity at 64 rows that only gets used at 256.

The persistent path splits a single row across CTAs and can use the whole GPU. So the question:

**Was `num_rows > 32 && smem >= 128 KiB → FilteredTopK` chosen because the filtered kernel is
genuinely better at those shapes, or because the persistent kernel of the day was slow there?** If it
is the latter, the condition may simply be stale — the deterministic persistent path in this PR is a
different kernel from the one that routing decision was made against.

Some evidence that it is not a one-way answer. At n=16,384 this branch's filtered path is *faster*
than its own persistent path at essentially the same work (17.9 µs at 33 rows vs 21.9 µs at 32 rows),
so "prefer persistent above 32 rows" would be wrong. Any fix has to be shape-dependent on `n`, not on
the row count alone.

I would like to measure the crossover — rows {48, 64, 128, 256} × n {16k, 20k, 40k, 65k} × k {512,
2048}, both paths at identical shapes on H100 — and if the persistent path wins at the long rows, the
change is a dispatch condition rather than a new algorithm. Before I do, two things I would rather
hear from whoever owns this code than guess at:

1. Is there a correctness or memory reason the persistent path is not entered with `num_rows > 32` on
   these parts? The workspace and `RadixRowState` sizing has never been exercised there, and
   `kDetMaxCtasPerGroup` is 64.
2. Is there history behind the 32 that I should not casually re-tune?

If the routing turns out to be sound, then the remaining cost is algorithmic and belongs inside the
filtered kernel — a wider coarse histogram plus an exact-refinement pass over the threshold bin,
rather than the four full-row radix passes it does today. That is a larger change than this PR should
carry, and I would rather do it as a follow-up than hold the correctness fix behind it. But the
routing question is cheap to answer and might remove most of the gap on its own, so it seems worth
asking first.
