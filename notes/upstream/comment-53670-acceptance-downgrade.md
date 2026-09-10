DRAFT — user go "which of the upstream post are really useful, if yes, post it". vllm#53670 (2026-09-10).

@Suppressor72 Your non-replication is right, and we have acted on it rather than defended the number.

**Our acceptance figure is downgraded.** We reported the trailing-block drop costing 4–6 pp of MTP
acceptance. You measured **+1.1 pp, Welch p = 0.59** on a different always-drafting layout and said
plainly that n=3 is underpowered to exclude a small effect. We have reclassified ours as
**configuration- or workload-dependent** rather than a property of the flag. What now stands on two
independent layouts is the throughput and prefix-hit half — your **+62 % warm throughput and +40.9 pp
hit rate**, our −26 % per warm turn.

**And we have a mechanism for why the acceptance halves disagree, which is worth more than the
disagreement.** On this box a **1-ulp** numerics change — a fused GDN kernel within one bf16 ulp of the
original, not a scheduling change at all — moved MTP acceptance **+6.6 / +10.4 / −3.8 pp across three
prompts**. Expectation zero, spread ±10 pp, from rounding alone. Any acceptance delta of single-digit pp
measured on one prompt set is inside that band. That covers your +1.1 and our 4–6 equally well, and it
means **neither of us should book an acceptance number for this flag without a per-prompt spread.**

So the question you put to maintainers is the right one and we would sharpen it only slightly: the
blanket drop has not bought measurable acceptance on either always-drafting layout tested, while costing
one alignment unit of recompute per prefix hit — and the acceptance side may not be measurable at all at
these sample sizes.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
