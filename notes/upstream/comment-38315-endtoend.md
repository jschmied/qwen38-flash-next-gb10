DRAFT — user go given ("yes post follow up"). vLLM PR #38315 follow-up (2026-09-09).

Follow-up to my kernel numbers above: I took the same port to a served endpoint, because "real but small at
the request level" deserved an actual request-level measurement rather than my extrapolation from layer count.

Served vLLM on one GB10 (sm_121, TP=1), Qwen3.8-Flash-Next NVFP4, MTP-3, prefix caching on, the fused kernel
behind an overlay so the two arms differ in nothing else, three interleaved server starts per arm:

| cell | two-kernel (×3 starts) | fused (×3 starts) |
| --- | --- | --- |
| cold TTFT, 7,528-token prompt | 3.187 / 3.196 / 3.321 s | **3.144 / 3.150 / 3.152 s** (−1.4 %) |
| cold TTFT, 29,288-token prompt | 12.116 / 12.207 / 12.215 s | **12.021 / 12.055 / 12.091 s** (−1.1 %) |
| 24 warm agent turns, paired total | 18.67 / 18.77 / 18.83 s | 18.74 / 18.67 / 19.58 s (null) |

The cold-prefill arms do not overlap on either cell, so the effect is real and reproducible — and it is
**~1 %**, not the 7–12 % the kernel itself gives. That is simply the GDN scan's share of a prefill on this
model, and it lands on the same side as @vadiklyutiy's B300 read: the kernel win is genuine, the request-level
win is small. On warm agent turns it is null. I would not want the earlier table read as an end-to-end claim.

One caution for anyone A/B-ing this with speculative decoding on, which cost me a confused hour. The fused
kernel matches the two-kernel path to within one bf16 ulp but is **not bit-identical**, and that is enough to
change which draft tokens the target accepts: across three prompts, with three starts each and acceptance
identical to the decimal within every arm, MTP acceptance moved **+6.6, +10.4 and −3.8 pp** — throughput
+5.2 %, +10.3 % and −2.7 %. Expectation zero; it is a lottery, not a speed effect. Acceptance is a speed
statistic here, not a quality one, and it should not be quoted for or against this change.

The branch (`jschmied/vllm:fla-fused-kkt-solve`, rebased onto the current
`vllm/third_party/flash_linear_attention/ops/` layout, with the natural-log gate and `FLA_TRIL_PRECISION`
adaptations and a test against the two-kernel path) is still there if it is useful to you.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
