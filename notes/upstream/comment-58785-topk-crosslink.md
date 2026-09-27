POSTED 2026-09-27 (user's go: "post"). vllm-project/vllm#58785 (persistent_topk overflow), cross-link to our #55122.

Cross-link: #55122 changes the same kernel and removes this failure by construction. It replaces the capped candidate buffers with a radix select that rescans the row for each key byte (at most five reads of the row), so there is no buffer to overflow; exact-key ties at the k-th value are ranked by index, which also makes the output deterministic. On a second GB10 it measured 21–28 % faster than the exact-topk workaround.

I have not run your new `test_persistent_topk_overflowing_threshold_bin` against #55122's kernel yet; I can do that on a GB10 if it helps decide between the two approaches. The two PRs will conflict, since both edit `persistent_topk.cuh` and `test_top_k_per_row.py`.

Written with AI assistance (Claude Code).
