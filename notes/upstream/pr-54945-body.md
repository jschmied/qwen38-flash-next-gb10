# PR body for #54945 — branch `fix/flashinfer-moe-fused-finalize` (local worktree ~/git/vllm-moe-finalize, not pushed)

Title: `[Kernel] Add VLLM_FLASHINFER_MOE_FUSED_FINALIZE to disable the nondeterministic MoE finalize`

---

## Purpose

Fixes #54945. `flashinfer.fused_moe.cutlass_fused_moe` has a `use_fused_finalize` argument whose default (fused epilogue, atomics) is documented as nondeterministic. vLLM's `flashinfer_cutlass_moe.py` never passed it, so identical requests at temperature 0 could return different logits on this backend with no switch to turn it off. This adds `VLLM_FLASHINFER_MOE_FUSED_FINALIZE` (default `1`, behaviour unchanged) and passes it through.

## Test Plan

- Qwen3.8-Flash-Next NVFP4 on a GB10 (sm_121), FlashInfer 0.6.17: one fixed prompt three times, `temperature=0`, `max_tokens=4`, `top_logprobs=20`, hashing every token's top-20 (script in #54945).
- Three serving shapes: prefix cache off + eager; prefix cache on (chunked prefill); prefix cache on + MTP n=5.

## Test Result

| shape | default | `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` |
| --- | --- | --- |
| cache off, eager | token 2 differs, 3 distinct of 3 | 1 distinct of 3 |
| cache on | token 1 differs, 3 distinct of 3 | 1 distinct of 3 |
| cache on + MTP n=5 | not probed | 1 distinct of 3 |

Decode cost of the unfused finalize: +3.6 % in one 8-turn agent-loop measurement (43.92 → 45.50 ms/tok; run-to-run band for this config 42.8–47.7), i.e. indicative. `ruff check` / `ruff format --check` clean.

Note for FlashInfer ≤ 0.6.17: with a populated `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, switching the finalize fails at init with `Invalid gemm2 profile id` because the autotune cache key did not distinguish the two runners; fixed in FlashInfer from v0.6.18rc2 (`MoERunner.get_cache_key_extras`). Use a fresh cache dir on older versions.

Includes AI-assisted code (Claude Code); the change and the repro were reviewed and run by me.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
