DRAFT — needs the user's go. vllm-project/vllm PR #55122 (our own PR), comment (2026-09-09).

@200lz — agreed on keeping `test_persistent_topk_degenerate_lengths`, and flashinfer-ai/flashinfer#5015
is a stronger argument for it than we had: a deterministic wedge on real traffic is worth a regression
test regardless of how the tie path behaves. Nothing below touches that; the degenerate-length case and
the tie case are separate paths and only the second is what we measured.

With that said — a measurement against our own PR, before a reviewer has to find it.

We instrumented `_topk` in the QSA indexer to census the top-k boundary on real serving traffic, asking
how often the kernel is actually in the position this PR makes deterministic — `n_gt < k < n_gt + n_eq`,
where it must pick only some of the entries tied at the k-th value and the choice is arrival-ordered.

GB10 / sm_121, TP 1, MTP 3, 16 k max len, `k = 512` blocks, 400 instrumented calls over two ~13.6 k-char
prompts:

| | |
| --- | --- |
| rows seen | 87,257 |
| selection was a no-op (`visible ≤ k`) | 81,065 (92.9 %) |
| rows that performed a real selection | 6,192 (7.1 %), across 267 of 400 calls |
| rows with **any** tie at the k-th value | **0** |
| rows ambiguous | **0** |

Not "tied but resolved consistently" — **never tied**. Exact ties between float32 logits out of a real
GEMM are rare, and on top of that most rows had fewer visible blocks than the budget, so the top-k
returned everything and order could not matter.

This agrees with an end-to-end isolation we ran separately. Taking a stock server that diverges on greedy
decoding (333 disagreeing positions on a 2.5 k-token prompt) and enabling one fix at a time: this PR alone
gives 330 — i.e. nothing. Three other fixes together with it give 0. So **we cannot claim this PR buys
end-to-end determinism on production traffic**, and we would rather say that ourselves than have it
assumed from the PR description.

What the PR still is, and we think still worth merging: the kernel is genuinely arrival-order dependent
under ties, which our standalone test shows directly with tie-heavy input, and the fix is cheap. It is
correctness under a condition that is rare on *this* workload rather than a throughput or reproducibility
win — and, per the comment above, the degenerate-length half of it guards something that has already bitten
a production deployment.

Scope of the null, stated plainly: 16 k context, two prompts, 6,192 selecting rows. Longer contexts push
a much larger fraction of rows past `visible > k` and would sample the boundary far more often, and a tie
rate indistinguishable from zero at ~6 × 10³ samples is not zero at 10⁶. If a workload with long contexts
and low-entropy logits is the interesting case for this kernel, we have the box and can census it — say
the shape you want and we will run it.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
