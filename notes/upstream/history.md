# Upstream history (pre-September)

<!-- moved out of README.md on 2026-09-05 so the README stays an entry point; content unchanged -->

## Upstream

| | |
|---|---|
| [#53896](https://github.com/vllm-project/vllm/pull/53896) | `[Model] Support Qwen3.8-Flash-Next` — **MERGED 2026-08-31**. Was for weeks the only place the vLLM implementation existed; it is now on `main`, though not yet in a tagged release. Note the package was renamed `qwen3_8_flash_next` → `qwen4_exp` before merge, so every source path we cite is from the pre-merge build. |
| [#50617](https://github.com/vllm-project/vllm/pull/50617) | fixes the `FP8_PER_CHANNEL_PER_TOKEN` dispatch gap we hit; we added our load-failure evidence rather than opening a duplicate |
| [#53899](https://github.com/vllm-project/vllm/pull/53899) | PLE offload to host memory. Carries `4e8b849b8d97`, the per-request event-pool fix for the [#53960](https://github.com/vllm-project/vllm/issues/53960) startup deadlock — **backported into our venv** ([notes](notes/upstream-branch.md)) |
| [#53960](https://github.com/vllm-project/vllm/issues/53960) | PLE offload deadlocks at warmup (4 reporters). We could not reproduce it and posted the mechanism, the fix pointer and a negative control |
| [#52816](https://github.com/vllm-project/vllm/pull/52816) | DFlash2 — **merged** 2026-08-21 |
| [our branch](https://github.com/jschmied/vllm/tree/gb10-sm121-fixes) | three commits on #53896's head: `quant_config` through `GatedResidual`, `quant_config` on both `ParallelLMHead` sites, and the dispatch fix |

Two contributions from earlier that still stand: a **one-line gate change** so the FP8 PLE is
accepted on an NVFP4 body (correcting checkpoint tables that list RadixArk as not loading — it
does), and **`--cap-add=SYS_PTRACE`** for `VLLM_PLE_CPU_OFFLOAD` in Docker, where
`rebuild_cuda_tensor` needs `pidfd_getfd` and the engine otherwise dies ten minutes in with only
`Failed core proc(s): {}`.

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.
