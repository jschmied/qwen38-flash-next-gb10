End-to-end numbers on GB10 (sm_121) for this PR's kernel, three server starts per arm, Qwen3.8-Flash-Next (QSA top-k on 8,192-row scores, ModelOpt NVFP4/FP8 checkpoint), vLLM preview build, MTP n=5, prefix caching on:

| | stock (a / b / c) | this PR (a / b / c) |
| --- | --- | --- |
| TTFT 7.5k tokens, median of 3 | 3.10 / 3.29 / 3.10 s | 3.12 / 3.35 / 3.12 s |
| TTFT 29k tokens, median of 3 | 11.24 / 11.55 / 11.26 s | 11.22 / 11.62 / 11.27 s |
| 8-turn agent loop, s per turn | 2.70 / 2.68 / 2.68 | 2.69 / 2.54 / 2.66 |

Every pair is inside the start-to-start band, i.e. no measurable TTFT or per-turn cost at the server level. A third arm that forces the exact selection through `torch.topk` produced byte-for-byte the same 8-turn run as the PR's kernel on all six starts (166 tokens, 65 drafts, 111 accepted), so the selected set is the exact one, and it is stable across restarts; the stock arm's runs differ from each other and from those (208–224 tokens).
