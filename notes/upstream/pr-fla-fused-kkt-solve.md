# DRAFT — vLLM PR: [Kernel][GDN] Port fla-core's fused kkt+solve kernel into the vendored FLA chunked forward (open on go)

Branch: `jschmied/vllm:fla-fused-kkt-solve` (3 commits, signed off).

## Summary

The vendored flash-linear-attention forward computes the per-chunk `(I + tril(beta·K Kᵀ))⁻¹` with two kernels,
`chunk_scaled_dot_kkt_fwd` + `solve_tril`. fla-core 0.5.2 fuses both into `chunk_gated_delta_rule_fwd_kkt_solve_kernel`
for the 64-token chunk. This ports that kernel (Apache-2.0, Songlin Yang / Yu Zhang, credited in the file header) as
`ops/chunk_kkt_solve.py` and dispatches to it when `FLA_CHUNK_SIZE == 64`; other chunk sizes keep the two-kernel path.

Two adaptations to the vendored conventions: the gate stays in the natural-log domain (`exp`, not fla-core's
`1/ln2`-scaled `exp2`), and the solve's dot precision reads the same `FLA_TRIL_PRECISION` knob as `solve_tril.py`
(default `ieee`) instead of fla-core's hard-wired tf32.

## Correctness

Fused A vs two-kernel A: max |Δ| 1.95e-3 (¼ bf16 ulp), mean 1.2e-5. Full `chunk_gated_delta_rule` output: max |Δ| one
bf16 ulp (7.8e-3 at |o| ≈ 1.4), final state max |Δ| 4.6e-3, at T = 333 / 2048 / 5000 / 7503 / 16384, single and
varlen (`cu_seqlens`), H = 16 key heads, HV = 48 value heads, K = V = 128 (Qwen3.8-Flash-Next's GDN shape).
Test added: `tests/kernels/mamba/test_fla_fused_kkt_solve.py`.

## Performance (GB10 / sm_121, bf16)

| T | kkt+solve two kernels | fused | whole chunk_gated_delta_rule |
| --- | --- | --- | --- |
| 2048 | 394 µs | 189 µs (2.1×) | 2,144 → 1,892 µs (−11.8 %) |
| 7503 | 1,466 µs | 799 µs (1.8×) | 7,677 → 6,949 µs (−9.5 %) |
| 16384 | 3,274 µs | 1,823 µs (1.8×) | 18,315 → 16,965 µs (−7.4 %) |
| 29263 | 6,814 µs | 3,687 µs (1.8×) | 35,018 → 31,961 µs (−8.7 %) |

The `h`, `o` and `w_u` kernels are untouched; the whole gain is the fused step. In a Qwen3.8-Flash-Next prefill
(36 GDN layers) this is ~1.5–2 % of TTFT.

_Written with AI assistance (Claude Code); every line reviewed by the author._
