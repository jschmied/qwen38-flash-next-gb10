POSTED 2026-09-22 — vllm-project/vllm PR #58157
<https://github.com/vllm-project/vllm/pull/58157>

## Purpose

The DeepGEMM E8M0 accuracy guard added in #38083 (fixing #37804) is keyed on **model type**:
`should_auto_disable_deep_gemm()` tests `_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES`, currently
`{"qwen3_5_text", "qwen3_5_moe_text"}`.

The precision loss it guards against is a property of the **quantization path**, not of the
architecture. It occurs whenever a checkpoint's float32 block scales are requantized to UE8M0 at
load time — a *second* rounding on top of the one already baked into the checkpoint. A model is
therefore protected only if someone has already measured a regression on it and added its
`model_type`. Everything else is requantized silently, with no signal at all.

#48302 shows the failure mode in practice: a new model adding *itself* to the exclusion set, having
learned about the problem from `sgl-project/sglang#30275` rather than from vLLM.

A concrete uncovered case, on hardware: **Qwen3.8-Flash-Next** (`model_type: qwen4_exp` /
`qwen4_exp_text`, ModelOpt `MIXED_PRECISION` — **157 `FP8_PB_WO` blockwise-FP8 layers** plus an
NVFP4 MoE) on a **DGX Spark GB10, sm_121**, vLLM `0.28.1rc1.dev524`. It is not in the exclusion set,
and with DeepGEMM enabled every blockwise linear selects the DeepGEMM kernel:

```
deep_gemm.py:136  DeepGEMM E8M0 enabled on current platform.
__init__.py:698   Selected DeepGemmFp8BlockScaledMMKernel for MergedColumnParallelLinear
__init__.py:698   Selected DeepGemmFp8BlockScaledMMKernel for QKVParallelLinear
__init__.py:698   Selected DeepGemmFp8BlockScaledMMKernel for RowParallelLinear
__init__.py:698   Selected DeepGemmFp8BlockScaledMMKernel for ParallelLMHead
```

— with nothing indicating its weights were requantized. #37804 measured **−12pp GSM8K** from this
path on Qwen3.5-FP8 across ~80 such layers; this checkpoint has 157. **We have not measured accuracy
for this model**, and this PR makes no claim about it. The point is that nothing tells a user to
look.

## What this changes

Nothing about policy or kernel selection. One `logger.warning_once` at the exact point where the
requantization runs, in `deepgemm_post_process_fp8_weight_block`, plus a test.

## Why a warning rather than a scheme-keyed auto-disable

The precise predicate for "this checkpoint loses precision" is already available per tensor:
`ws.dtype == torch.float32 and use_e8m0`. Checkpoints that ship E8M0 scales
(`float8_e8m0fnu` / `uint8`) take the other branch, are not requantized, and lose nothing — so
"DeepGEMM is on" alone is too coarse a condition.

Auto-disabling on that predicate would however be too broad: DeepSeek FP8 checkpoints ship float32
`weight_scale_inv` and are DeepGEMM's primary target, so it would turn DeepGEMM off for the workload
it was written for. A warning costs no throughput, cannot regress anyone, and addresses the actual
failure mode, which is silence. If maintainers prefer something stronger, this is the natural place
to hang a config-gated disable in a follow-up.

## Test Plan

```
pytest tests/quantization/test_fp8.py -k ue8m0_requant_warns -q
```

`test_deepgemm_ue8m0_requant_warns_only_when_it_runs[True|False]` calls
`deepgemm_post_process_fp8_weight_block` with float32 block scales and asserts the warning appears
iff `use_e8m0` is set. `warning_once` dedupes through an `lru_cache`, so the test clears
`_print_warning_once.cache_clear()` first — without that the second parametrization passes
vacuously.

## Test Result

Run on GB10 sm_121 against a build carrying the patch: `2 passed, 14 warnings in 3.89s`, no
device-side assertions. The shipped test contrasts float32 vs `float8_e8m0fnu` scales at
`use_e8m0=True` (warn iff the requant runs). An earlier draft contrasted `use_e8m0=True/False` with
float32 scales; that trips `smxx_layout.cuh:131` when both cases share a process — the
float32-scales-on-SM12x hazard of #57512, unrelated to this change but not worth shipping.

---
Essential Elements checklist + AI-assistance disclosure to be added in the posted body.
