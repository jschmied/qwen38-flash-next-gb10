DRAFT — needs the user's go. Reply on #54521 to davidcanar (21:19).

Thanks, that is a clean cut, and the correction makes the result stronger rather than weaker: strictly sequential, idle, cache off, still 5/5 distinct means a reduction-order variable is in your path and nothing else needs to be true.

The recurring hashes are the most useful observation in the thread so far. A small discrete outcome set is exactly what a reduction with a handful of possible partial orders produces (two or three interleavings of the same partials), and it is hard to get from an out-of-bounds read, which would not repeat. With MoE combine and the ragged decode kernel excluded bitwise, that leaves the collective — `NCCL_PROTO=Simple` is the right next pin — and the kernel that just faulted.

That fault is worth its own issue regardless of this one: `--max-num-seqs 1` → illegal address in `_deepgemm_fp8_paged_mqa_logits` on the first request is a reproducible engine kill in the sparse indexer, and the same allocation-size dependence would read wrong values silently one page earlier. If you file it, link it here; I will add the GB10 side if the kernel is shared.

The gfx1151 rows are a genuinely different picture: BF16 M-invariant through M = 64 and the tuned int4 MoE bit-identical at every M. Please open the PR for `tools/gemm_m_invariance_rocm.py` — same structure, a `--backend` note in the docstring, and I will add your rows next to the sm_121 ones in the README table. The two-node offer stands if the collective turns out to be the channel; that comparison is worth doing properly.
