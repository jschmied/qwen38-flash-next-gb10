DRAFT — needs the user's go. vLLM PR #55122 comment (2026-09-06 20:5x).

Two commits pushed.

**`c564e5c` — the host guard rejected short rows.** The `chunk_size >= TopK` check I added with the
kernel is unconditional, but only the cooperative large path (`max_seq_len > RADIX_THRESHOLD`) ranks
the final candidates inside CTA 0's chunk buffer and needs it. Rows at or below the threshold take
the single-CTA select, or the trivial `seq_len <= TopK` case, and never touch that buffer.

That shape is not hypothetical: the block-level QSA indexer calls `persistent_topk` with
`TopK = token_topk / compress_ratio = 512` over a few hundred blocks at warm-up, so on a build that
takes that path the server died at start with

```
persistent_topk: chunk_size 256 smaller than TopK 512
```

The guard is now conditional on the path, and the message names it.

**New regression test** `test_persistent_topk_short_rows`: rows of 256–2048 at TopK 512/1024/2048,
33 cases. The exactness matrix skipped every one of these, because it required `top_k < seq_len`.
The reference now pads the unused slots with `-1`, which is what the kernel writes there.

On a GB10 (sm_121, TP1, torch 2.13 / CUDA 13), this PR's kernel built standalone and driven by the
PR's own test file:

| | short-row cases | whole `persistent_topk_` selection |
| --- | --- | --- |
| with `c564e5c` | 33 passed | 134 passed, 26 skipped |
| without it | **24 failed**, 9 passed | — |

The 9 that pass without the fix are the `seq_len == top_k` cells, where `chunk_size` happens to
satisfy the old check.

**`afd9281` — the review round from 03 Sep.** Per-call device properties (the static cache was shared
across devices, and the dynamic-smem cap depends on the device the launch runs on) and `k=1024` in
the exactness matrix. I committed this locally on the 3rd and reported it here, but never pushed it;
it is on the branch now. Apologies to anyone who read the diff in between.

`pre-run-check` fails on the label policy, not on the code; format checks (clang-format, ruff) are
clean locally.

*AI assistance was used for this change; every line was reviewed by me before pushing.*
