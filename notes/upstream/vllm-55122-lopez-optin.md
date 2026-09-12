POSTED 2026-09-12 (user go "post all"). vllm#55122 reply to @LopezCastroRoberto.
---
Sorry for the slow reply. On the substance I think you are right, and our own data supports your
position rather than mine.

**The defect this PR fixes is unreachable on the traffic we censused.** Of 6,192 rows that actually
performed a selection, **0 had any tie at the k-th value** (93 % of rows had fewer visible blocks
than k, so selection was a no-op). Stated with its bound, because it matters: that census was at
**16k context over two prompts**, so it says nothing about the long-context regime where
[#51782](https://github.com/vllm-project/vllm/issues/51782) is now seeing trouble. Within that
bound, what this PR buys is kernel correctness under ties rather than end-to-end determinism — our
framing to own, and it argues against changing a default.

**We have effectively been running your design for a week.** Our deployment carries the
deterministic kernel out-of-tree behind an env flag (`VLLM_QSA_DET_TOPK`), opt-in, default off. That
is the shape you are proposing.

**One more argument against defaulting it, from this week.** The deterministic path wants the row
resident in shared memory. On GB10 (`sharedMemPerBlockOptin` = 101,376 B) that request exceeded the
device at ~100k context and hard-failed:

```
persistent_topk_det: dynamic smem 98080 exceeds 97120 (optin 101376 - static 4256)
```

The failure to clamp is in *our* out-of-tree wrapper and is ours to fix — but the appetite is the
deterministic algorithm's, and a low-shared-memory device cannot always satisfy it. That is a reason
for opt-in, not for a default.

**Yes, we can test #55872 here** and will report back. It is pure Python and it touches our exact
path (`models/qwen4_exp/nvidia/indexer_qsa.py`, `ops/qsa_indexer.py`), so it is straightforward on
one GB10, sm_121, TP1, Qwen3.8-Flash-Next NVFP4.

One scoping note that may matter for where the backend hooks in: on GB10 the stock path is not
`persistent_topk` for every row. Leaving it for `top_k_per_row_decode` needs three conditions
together — row length above `RADIX_THRESHOLD`, the cooperative launch oversubscribing, and
`sharedMemPerBlockOptin < 128 KiB`. GB10 always meets the third, so short rows and long rows here run
different kernels. A backend that wraps only the `persistent_topk` selection would cover the short-row
case on this hardware and miss the long-row one.
