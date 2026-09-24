DRAFT — needs the user's go. GitHub vllm-project/vllm PR #58449, GB10 server A/B (2026-09-24).

Server A/B of this PR on one DGX Spark (GB10, sm_121, TP=1): vLLM main `1ea7c63f4` with only `qsa_cache.py` from
`848dbf89e` swapped in, Qwen3.8-Flash-Next NVFP4, MTP n=3, PIECEWISE, 3 interleaved starts per arm.

- **The new test passes on sm_121:** `-k draft_decode_metadata_update` is 3/3 with the PR's `qsa_cache.py` and
  3/3 failing with main's, so it does exercise the change.
- **It engages.** The `Fused multi-step draft decode is not supported … QWEN4_EXP_EXP_QSA_STATE` line is in every
  baseline log and in none with the PR.
- **Output is bit-identical.** Greedy output hashes (4 prompts at c=1 and 4 at c=4) and acceptance counters match
  exactly across all six starts.
- **Speed: no measurable change on this box.**

| | c=1 decode ms/tok | c=4 aggregate tok/s |
|---|---|---|
| main | 24.057–24.351 | 83.72–85.25 |
| this PR | 24.148–24.229 | 84.00–85.29 |

This is not an argument against merging it: it is correct, and a host-bound setup, such as higher concurrency or
FULL graphs for the drafter, may well see the saving. On GB10 at c ≤ 4, decode is GPU-bound and async scheduling
already hides the metadata rebuild.

*AI assistance was used in preparing this comment; the measurements were run and checked by me.*
