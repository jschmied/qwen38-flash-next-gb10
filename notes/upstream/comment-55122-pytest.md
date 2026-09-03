Test status on a GB10 (sm_121), against a standalone build of this PR's exact sources (the box runs the preview image, so `_C` itself is the stock kernel; a pytest plugin swaps the op):

- the three new test functions in this PR: **70 passed**, 8 skipped (k ≥ n);
- the same tests against the unmodified `persistent_topk`: **70 failed** (every case, tie-free random rows included — the stock output order is neither reproducible nor index-sorted).

End-to-end TTFT / decode A/B (stock vs this kernel vs exact `torch.topk`, 3 starts each) runs tonight; numbers follow.
