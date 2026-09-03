# Main-tree venv (`/opt/llm/runtime/vllm-venv-fnmain`)

Built 2026-09-03 from the nightly aarch64 wheel `vllm-0.28.1rc1.dev352+gbb363db9a` (torch 2.13.0 pin
satisfied by the cloned venv's `2.13.0+cu130`; flashinfer left at 0.6.17, the 0.6.18 pin is unmet —
0.6.18 drops the SM121a JIT cubins). `build-fnmain.sh` is the clone-don't-build recipe.

What main has that the preview lacks: the sm120 blockwise-FP8 dispatch fix (#52775, finding 65),
the prefill/decode-split QSA indexer (#54513), FP8_PB_WO natively, the `qwen4_exp` model package.
What main lacks: the PLE CPU-offload worker (`VLLM_PLE_CPU_OFFLOAD`) — replaced here by generic UVA
offload of the n-gram tables (`--cpu-offload-gb 56 --cpu-offload-params ngram_embedding`, in
`/opt/llm/serve-fnmain.sh`).

Two loader patches, env-gated scripts with `off`:
- `ple_gate_patch.py` — the PLE FP8 embedding method for ModelOpt (mixed/fp4) checkpoints (port of
  the 2026-08-27 preview patch; main's gate accepts only `Fp8Config`).
- `scaleinv_patch.py` — `weight_scale_inv` rank-2 → `weight_scale` [ob,1,ib,1] at load time (ModelOpt
  0.46 FP8_PB_WO export convention; defects 3+4 of `flashnext-fp8mix-checkpoint`).

Both revert on any reinstall. Memory: body 69.4 GiB on the GPU + 47.7 GiB PLE pinned = 117 GiB of 121;
serve with `FN_UTIL` ≈ 0.60 and `--language-model-only`.
