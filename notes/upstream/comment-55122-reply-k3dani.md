Thanks for the independent run — the 99–100 % vs 87–90 % throughput split matches what the microbenchmark predicted.

On the ordering point: both selections return the same *set*; this kernel emits it in ascending index order, `torch.topk` in descending value order, and the sparse attention sums in output order, so the two can differ at the last ulp. Each is reproducible with itself; neither is "more correct". If your KIE suite shows a difference between them, that is a tie-break/summation-order effect, not a selection error — happy to see the numbers either way.

Our own before/after on the same kernel (TTFT 8k/30k and MTP decode, 3 starts each) follows here once the box is free.
