DRAFT — needs the user's go. vLLM PR #55122 review request (2026-09-06 21:0x).

@mgoin @tlrmchlsmth — could I ask one of you for a look at this when you have a moment? It has been
open since 03 Sep and is ready from my side.

What it does: `persistent_topk` handed out output slots with `atomicAdd` in thread-arrival order and
took exact-key ties first-come, so the selected order — and, when more keys tie at the threshold
than the candidate buffers hold, the selected set — varied between identical calls. Sparse attention
sums the selected keys in output order, so greedy decoding forked between identical requests. The
kernel now emits in index order from an exact pivot, with ties ranked by index. It also fixes #51782
(dropped candidates), and closes one of the three causes behind #54521.

Evidence in the thread: 134 kernel test cases on a GB10 (sm_121), an end-to-end server A/B showing
no TTFT and no per-turn cost, and an independent validation by @k3dani on another GB10.

The only red check is `pre-run-check`, which gates on a `ready`/`verified` label or four merged PRs
by the author; `Check format` and `pre-commit` are skipped behind it and have not run. clang-format
and ruff are clean locally.
