Rebased onto `b6e2aa748b`; 15 commits, conflict-free. The only conflict was in
`tests/kernels/test_top_k_per_row.py` against #56464 and was purely additive on both sides — no
overlapping names, both test sets kept. Neither kernel source moved upstream: `persistent_topk.cuh`
and `topk.cu` diff byte-for-byte identically to before the rebase, and the five commits that
disappeared were earlier `Merge branch 'main'` syncs.

For anyone reading after #56464: `persistent_topk` is still live as the `persistent` backend in the
new `indexer_topk.py` dispatcher, so this fix now applies to a path selected explicitly.

@MaCoredroid — thank you, and sorry for the slow acknowledgement. Driving `force_single_cta` /
uncached `det_select_row` directly at 48 SMs and 101,376 B opt-in shared memory is the case we could
argue for but not demonstrate; 324 fallback launches matching a value-descending / index-ascending
reference across six repeats each, plus the cooperative-control launches at 355584 and the expected
>64-CTA rejections at 474116, says more about the fallback than anything in the PR body. Publishing
the harness and source hashes is what makes it checkable, and your scope note is right — one device,
kernel level, nothing about operator registration, CUDA graphs, other streams or performance.

With @k3dani's serving-level run that is two independent GB10 confirmations from different angles.
