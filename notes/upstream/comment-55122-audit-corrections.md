DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122, corrections to two of our comments (2026-09-24).

Two corrections to my own comments here, found while auditing our notes.

**1. My perf comment from today** ([link](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5809971218))
said @k3dani's 21–28 % was explained by their workaround being a full-row sort. That was too strong.
- Their full-sort workaround is real: mode 1 of `qsa_exact_topk.patch`.
- But in the 09-12 run the two arms also differed in image and vLLM version: the preview `0.1.dev20073` against
  the official `v0.29.0` (their eval-card).
- On a single image, in their 09-03 run, the PR beat the same workaround by 10–15 %: 2253 vs 1952 tok/s at
  6,082 tokens, and 1932 vs 1756 at 24,416.
- So the sort explains part of the gap, and the rest can't be attributed. The rest of that comment stands: the
  kernel is faster than its merge base on GPU time, TTFT is the same as stock, and exact `torch.topk` costs 3–4 %
  at 30k.

**2. My #55872 result from 09-12** ([link](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5645230534))
said I set `VLLM_QSA_DET_TOPK=0` so that our kernel could not mask the FlashInfer path. It didn't do that: our
overlay treats any non-empty value as on, so the `native` arm ran our deterministic kernel. The finding that
matters is unaffected, since the FlashInfer backend still fails at engine init on sm_121. But "native 8/12
matches our unpatched baseline" was not a stock measurement.

*AI assistance was used in preparing this comment; the corrections were checked against the raw data by me.*
