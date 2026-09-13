DRAFT — needs the user's go. vllm-project/vllm PR #55122 comment (2026-09-13 ~20:40).

Rebased onto `b6e2aa748b`; the branch is now 15 commits and conflict-free. The only conflict was in
`tests/kernels/test_top_k_per_row.py` against #56464, and it was purely additive on both sides — the
two test sets have no overlapping names, so both are kept. Neither kernel source moved upstream:
`csrc/libtorch_stable/persistent_topk.cuh` and `topk.cu` diff byte-for-byte identically to before the
rebase. The five commits that disappeared were earlier `Merge branch 'main'` syncs.

Worth noting for anyone reading this after #56464: `persistent_topk` is still live. It is the
`persistent` backend in the new `vllm/model_executor/layers/indexer_topk.py` dispatcher, so this fix
now applies to a path a user selects explicitly rather than one they land on by default.

@MaCoredroid — thank you, and sorry for the slow acknowledgement. Driving `force_single_cta` /
uncached `det_select_row` directly at 48 SMs and 101,376 B opt-in shared memory is the case we could
argue for but not otherwise demonstrate, and 324 fallback launches matching a value-descending /
index-ascending reference across six repeats each, plus the 108 cooperative-control launches at
355584 and the 18 expected >64-CTA rejections at 474116, is a stronger statement about the fallback
than anything in the PR body. Publishing the harness and the source hashes is what makes it
checkable. Your scope note is right and I would not stretch it further: it is one device, kernel
level, and says nothing about operator registration, CUDA graphs, other streams or performance.

With @k3dani's serving-level run this now has two independent GB10 confirmations from different
angles — one end-to-end, one kernel-level on the overflow path.
