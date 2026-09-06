# POSTED 2026-09-06 (user go "post to 38315"): https://github.com/vllm-project/vllm/pull/38315#issuecomment-5559027143

Data point from a different part, in case it changes the calculus here: on GB10 (sm_121, one Blackwell-family SoC with
273 GB/s LPDDR5X, no FlashInfer GDN path) the fused kkt+solve is not within measurement error.

Qwen3.8-Flash-Next shape (H = 16 key heads, HV = 48 value heads, K = V = 128, bf16, chunk 64), kernel time of the whole
`chunk_gated_delta_rule` forward, two-kernel vs fused, 3 starts each:

| T | kkt+solve (two kernels) | fused | whole forward |
| --- | --- | --- | --- |
| 2048 | 394 µs | 189 µs | 2,144 → 1,892 µs (−11.8 %) |
| 7503 | 1,466 µs | 799 µs | 7,677 → 6,949 µs (−9.5 %) |
| 16384 | 3,274 µs | 1,823 µs | 18,315 → 16,965 µs (−7.4 %) |
| 29263 | 6,814 µs | 3,687 µs | 35,018 → 31,961 µs (−8.7 %) |

The `h`, `o` and `w_u` kernels are untouched; the gain is entirely the fused step. Across the model's 36 GDN layers that
is ~1.5–2 % of prefill time, so small at the request level, but it is real and free.

On accuracy (the earlier report in this thread): fused `A` vs the two-kernel `A` differs by at most 1.95e-3 (a quarter
bf16 ulp), and the full forward output by one bf16 ulp, at T = 333 / 2048 / 5000 / 7503 / 16384, single and varlen. Two
details matter for that: the vendored gate is in the natural-log domain (fla-core's kernel uses `exp2` with a `1/ln2`
scale — porting it unchanged gives a 5e-2 error), and the solve's dot precision should follow the vendored
`FLA_TRIL_PRECISION` knob (default ieee) rather than fla-core's hard-wired tf32.

I have the same kernel rebased onto the current `vllm/third_party/flash_linear_attention/ops/` layout with those two
adaptations and a test against the two-kernel path (`jschmied/vllm:fla-fused-kkt-solve`). Happy to fold it into this PR
or open it separately if this one is closed as stale — whichever the maintainers prefer.
