DRAFT — user go given ("post on #51782"). vLLM issue #51782 (2026-09-07).

@Leonccaa the third path you identified is worth a note, because a kernel-side fix for this is open as
#55122 and that path is **bypassed** there rather than patched: `histogram_2048_topk` has **zero callers**
in that PR. Every row at or below `RADIX_THRESHOLD` goes through a rescanning radix select instead, so the
`DBUF = 3708` truncation you found never runs. Same for the other two candidate buffers — the select
rescans the row per key byte, so there is no buffer to overflow and the pivot is exact; ties at the pivot
are then ranked by index rather than by arrival, which is the second half of the bug (the *set* can be
wrong when more keys tie than fit, and the *order* is wrong always).

We reached this from the same direction you did — a DSA-style model, in our case Qwen3.8-Flash-Next on a
GB10. Greedy decoding forked between byte-identical requests, and `persistent_topk` was one of three causes
(the others were a non-deterministic NVFP4 MoE finalize, #54948, and GDN align-block seeding, #54076). With
all three fixed, strictly sequential greedy reproduces bit-for-bit, 6 of 6 across server restarts. Your
`rows=1` observation matches ours exactly: the decode shape is affected, not just large batches.

On the accuracy question raised in #53287 — that no measurable regression shows up in MAIN, so overflow may
be rare or harmless: we have a case where it is not harmless, though it is not an accuracy metric. The
failure we care about is **reproducibility**: the sparse attention sums the selected keys *in output order*,
so an order-only change is enough to fork the hidden state, with no overflow required. Whether that matters
depends entirely on whether a deployment needs identical output for identical input. For us it did, and it
also broke A/B measurement, since two arms could not be compared.

On cost, since that is the usual objection: measured on GB10, the current head of #55122 is **1.00–2.45× the
unmodified kernel, with 14 of 43 shapes at or below it**, and a server-level A/B over three starts per arm
shows no TTFT change and no per-turn cost. @xueyangcs, if HPC-Ops is 2× faster while remaining exact, that
is a stronger starting point than anything in-tree — the property to check first is whether it is also
*order*-deterministic, not only set-exact, for the reason above.

Happy to share the standalone harness we use (bit-identity across repeats plus equality to an exact
reference under tie-heavy and all-equal inputs, 142 cases). It is not sm_121-specific and should build
against a V100 tree unchanged, if that is useful for checking your patched copy.

One practical note: `/csrc/libtorch_stable` has no `CODEOWNERS` entry, so #55122 currently has no assigned
reviewer. If anyone here has an opinion on the kernel, that thread is the place where it would help most.

*AI assistance was used in preparing this comment; the measurements are ours.*
