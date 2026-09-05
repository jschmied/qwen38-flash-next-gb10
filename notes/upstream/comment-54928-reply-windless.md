DRAFT — ON HOLD 2026-09-05: the blockwise-FP8 row below is WITHDRAWN (finding 122, scale-layout artifact); rewrite before any post.

Thanks — that is diagnostic 2 from above with the attribution done properly, and it lands where we ended up on the GB10: the drafter is exposing a batch-shape dependence of the target, E == V ≠ A at sub-0.13-nat margins.

On our box the same signature had three kernel causes (#54945/#54948, #54076/#53798, #55122); none applies to your stack (FP8 dense, no prefix cache, sm_120), so the next candidate is the GEMM. The sm120 CUTLASS FP8 dispatch picks its tile shape (and a swap-AB kernel for small M) by M, and so does cuBLAS, so a decode row at M=1 and the same row inside an M=8 verification block do not have to be bit-identical, and 0.1 nats is well within what a different accumulation order does to a logit.

Measured on a GB10 (sm_121, stock nightly `_C`): the same first row through M = 1, 2, 3, 4, 8, 9, 16, …, 4096 on the 27B's dense shapes (12288×2560, 5120×5120, 16384×2560):

| path | row 0 identical to the M=1 result | max \|diff\| (bf16 out) |
| --- | --- | --- |
| per-channel FP8 (`cutlass_scaled_mm`, sm120) | every M | 0 |
| blockwise FP8 128×128 (`cutlass_scaled_mm`, sm120 blockwise) | M=1 only — differs from M=2 on | 1.0–1.5e-2 |
| BF16 cuBLAS | M=1 only — differs from M=2 on | 0.25–2.0 (unscaled randn) |

`Qwen/Qwen3.8-27B-FP8` is blockwise, so on your stack every dense projection of every layer computes the verified row with different arithmetic than the decode row. That is a candidate rather than an attribution (GEMM alone, not logits), but it is enough to produce 0.1-nat flips after 40+ layers, and it matches your two other observations: `--enforce-eager` cannot change a kernel's tile choice, and batch shape (`--max-num-seqs`, a concurrent request) changes M the same way. Script: https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/tools/gemm_m_invariance.py (needs only a vLLM venv with `_C`).

If that reproduces on your card (the script needs only a vLLM venv with `_C`), the fix is a kernel-selection pin for M ≤ max spec block, not anything in DFlash2 — the `VLLM_BATCH_INVARIANT` machinery does exactly that for the attention/GEMM kernels it covers, and GDN is the gap.
