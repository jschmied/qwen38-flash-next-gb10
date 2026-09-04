DRAFT — post only after the user's go and after `gemminv` has a result (fill the bracketed line).

Thanks — that is diagnostic 2 from above with the attribution done properly, and it lands where we ended up on the GB10: the drafter is exposing a batch-shape dependence of the target, E == V ≠ A at sub-0.13-nat margins.

On our box the same signature had three kernel causes (#54945/#54948, #54076/#53798, #55122); none applies to your stack (FP8 dense, no prefix cache, sm_120), so the next candidate is the GEMM. The sm120 CUTLASS FP8 dispatch picks its tile shape (and a swap-AB kernel for small M) by M, and so does cuBLAS, so a decode row at M=1 and the same row inside an M=8 verification block do not have to be bit-identical, and 0.1 nats is well within what a different accumulation order does to a logit.

[GEMM result on sm_121, `tools/gemm_m_invariance.py`: per-channel FP8 / blockwise FP8 / BF16 cuBLAS row-0 identity across M=1…4096 on the 27B's dense shapes — TO FILL]

If that reproduces on your card (the script needs only a vLLM venv with `_C`), the fix is a kernel-selection pin for M ≤ max spec block, not anything in DFlash2 — the `VLLM_BATCH_INVARIANT` machinery does exactly that for the attention/GEMM kernels it covers, and GDN is the gap.
