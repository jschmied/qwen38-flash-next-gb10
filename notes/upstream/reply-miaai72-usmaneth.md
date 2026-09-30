POSTED 2026-09-30 (user's go: "do all important replies"). MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#72, reply to @usmaneth's review comment on README.md:1212.

@usmaneth I wrote the deterministic kernel this README section refers to, so here's data on the question. I ran stock `torch.ops._C.persistent_topk` and our deterministic `_C_det` kernel at the QSA widths (k = 512, 1024, 2048): 1,152 rows with visible lengths from 1 to 3k, 916 of them underfull, with random and tied logits, on GB10 in the vLLM `1ea7c63f4` build.

- **A `-1` before a real id:** 0 rows in either kernel. The `-1` padding was always a tail. So on these shapes the sort never pulls a block in from past a `-1`, which is the case you describe.
- **Order of the real ids:** stock returned them ascending in 454 of 576 rows and in another order in 122 (21 %). The deterministic kernel is always ascending, which is its contract and is tested against an exact reference, including rows shorter than k. So the sort does reorder real ids on about a fifth of rows.

That leaves one open question: can `_expand_qsa_indices_kernel`'s cap (`complete_blocks * COMPRESS_RATIO`) ever be smaller than the number of real ids on a row? If it can't, as it shouldn't when `visible_blocks` and `complete_blocks` come from the same position, order never decides what survives. If it can, the order would decide it with or without the sort, and stock's order is not stable across runs. I haven't instrumented the expand kernel to settle it. The test script is `tools/determinism/topk_minus1_order.py` in jschmied/qwen38-flash-next-gb10.

Written with AI assistance (Claude Code).
