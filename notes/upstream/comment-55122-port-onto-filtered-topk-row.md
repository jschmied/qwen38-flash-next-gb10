Status after looking at the rebase mergify has been asking for: **this is a port, not a rebase**, and I have it building but not yet validated.

**Why it is not mechanical.** Upstream split `FilteredTopKUnifiedKernel` into a thin `__global__` wrapper plus a `__device__` helper:

```
__device__ bool filtered_topk_row(score, dst, length, top_k, FilteredTopKStorage<MAX_K>&)
template <..., bool CheckOverflow = false>   // false => caller retries with exact full-row selection
```

This PR replaces the body of the function that existed before the split, so `git rebase` offers the helper as the conflict. Taking either side compiles and is silently wrong.

**Where the determinism belongs: the wrapper, not the helper.** `filtered_topk_row` has a second consumer — `sampled_topk.cuh` calls it with `CheckOverflow=true` for rows below `kMinSampledLength` and falls back to an exact threshold+emit when it returns false. Making the helper deterministic would change a kernel this PR is not about, and would make that fallback unreachable, because the rescanning select never stashes and so can never overflow. Putting the change in the wrapper keeps the blast radius exactly where it was before the split, and `sampled_topk.cuh` is untouched.

The kernel itself is unchanged: `det_select_row` is byte-identical to the pre-port branch. Only the call site moved.

**What is verified:** `nvcc -std=c++17 -arch=sm_121` compiles the header and generates code for both instantiations the launcher uses (VEC_SIZE 4 / predicated-loads false, and 1 / true), rc=0, no warnings. That compile caught a real bug in my own port — the three-way merge had dropped `#include "topk_histogram_4096.cuh"` and `namespace hist4096 = topk_histogram_4096;`, which the earlier revision no longer needed because it deleted the short path this port keeps.

**What is not verified, and I would rather say so than restate old numbers:**

1. `tests/kernels/test_top_k_per_row.py` has not run against the port. It needs a full `_C` build, which is queued here.
2. The 21–28% figure from the September GB10 runs predates 253 upstream commits, including the refactor above. The baseline it was measured against has moved, so I am treating it as unconfirmed until re-run.
3. My own tie census remains the weak point: 0 ties at the k-th value across 6,192 selecting rows, at 16k context over two prompts. That says nothing about long context, and no kernel benchmark answers it — it needs a serving run, which I cannot do at present.

I have not force-pushed. The port collapses the current 15 commits into a smaller set, so I would rather confirm that is wanted before rewriting what reviewers have been reading.

*AI assistance was used in preparing this comment; the changes and measurements referenced are ours and were reviewed before posting.*
