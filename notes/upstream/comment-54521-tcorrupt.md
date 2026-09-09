DRAFT — needs the user's go. GitHub vllm-project/vllm issue #54521, reply to mmastrac (2026-09-09).

@mmastrac Ran your repro. It reproduces here, harder than on your box, and the four fixes we have been
carrying close it completely.

**Cell:** Qwen3.8-Flash-Next NVFP4 on one GB10 (sm_121), **TP=1**, MTP-3, `max_model_len` 65536, your
script unmodified — 47 tools, the synthetic transcript at **49,902 prompt tokens** here, 40 runs at
temperature 0 with your fixed seed, prefix caching on.

| arm | distinct completions / 40 | runs diverging from the majority |
| --- | --- | --- |
| stock | **40** | 39 |
| all four determinism fixes on | **1** | 0 |

Your 4× GB10 TP=4 GLM-5.3-Flash run gave 5 distinct in 40 with the corruption at token 23–31. Ours
diverges at **token 0 or 1** — the opening token alternates between `I've`, `I keep`, `The` and `Let`
across runs. Different model, different TP, same hardware family, and the failure is not a rare
corrupted argument but total non-reproducibility from the first token.

The four fixes, for the record, because no single one of them does this on its own — we isolated them
one at a time against a no-fix control on a shorter prompt and got 333 disagreeing positions for stock,
330 / 334 / 285 / 280 for each fix alone, and **0** for all four together:

1. deterministic `persistent_topk` — [#55122](https://github.com/vllm-project/vllm/pull/55122), open
2. bit-stable NVFP4 MoE finalize — #54945 / #54948
3. the FlashInfer autotune **cache key** (without it the finalize fix will not even start: `Invalid gemm2 profile id`)
4. the PLE offload semaphore reset — [PR #13 on the #53899 branch](https://github.com/peakcrosser7/vllm/pull/13), **not upstream in any form**

A fifth, the PLE state-indices stride ([#55375](https://github.com/vllm-project/vllm/pull/55375)), is
already merged and was present in both arms.

Two things this says about the thread. **A single-fix A/B on a machine carrying more than one of these
will read as null** — which is a good way to discard a correct fix, and may be what several reporters
here are seeing. And your framing is the right one: the reproducibility number is the diagnostic, but
the cost is that a corrupted tool *name* loses the whole turn. On agent traffic that is worth more than
any throughput knob; the best speed lever we measured this week is 9.6 %.

Happy to run your script against other configurations on this box — TP=1 is the axis you do not have.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
