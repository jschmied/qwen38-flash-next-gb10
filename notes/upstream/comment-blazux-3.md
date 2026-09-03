A kernel-side fix for this is now a vLLM PR: vllm-project/vllm#55122 makes `persistent_topk` deterministic — index-ordered output, exact tie handling, no candidate buffers (so it also fixes vllm#51782, the dropped candidates).

Cost on GB10: 1.3–3× per call in the microbenchmark, ≈ +2 % TTFT at the model level by estimate, versus the −17…−40 % prefill that an exact `torch.topk` replacement costs (ours was +6 %). End-to-end decode/TTFT numbers follow on the PR.

If you want it on the preview tree without rebuilding vLLM: standalone build + env-gated wiring in https://github.com/jschmied/qwen38-flash-next-gb10/tree/main/patches/kernel-det (`build_det.py`, `tools/determinism/qsadet_patch.py`, `VLLM_QSA_DET_TOPK=1`).

For full reproducibility of this model on GB10 two more defects matter: the FlashInfer CUTLASS MoE fused finalize (vllm#54945, PR #54948) and the align-mode block-size PRs vllm#54076 / #53798.
