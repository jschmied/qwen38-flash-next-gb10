# The patch series, and where it can go

## There is no "main" to target

`vllm/models/qwen4_exp/` does not exist on vllm-project/vllm **main** — nor does anything
Flash-Next. Code search on the default branch returns zero for `qwen4_exp`,
`QSAMetadataBuilder`, `QWEN38NEXT_GEMM_PLANS`. No released version has it either (newest is
v0.28.0). The code lives only on PR #53896, which is open, conflicted, and 32 ahead / 49 behind
main.

Our serving venv is an unpacked wheel `0.1.dev20073+g8e685d198` taken from a vendor container;
that revision **does not exist in vllm-project/vllm** (HTTP 422). So every patch we hold is
written against a tree with no upstream identity, and the PR has since renamed the package
`qwen3_8_flash_next` → `qwen4_exp` and the dict `QWEN38NEXT_GEMM_PLANS` →
`QWEN4_EXP_GEMM_PLANS`.

## Branch

`gb10-sm121-fixes`, branched from PR #53896 head `82399a9`, in `~/vllm-fork`. Three commits,
12 insertions. Exported to `patches/series-pr53896/` and verified to apply cleanly to a pristine
PR head with `git am --3way`.

| commit | what | evidence |
| --- | --- | --- |
| `[Quantization] Dispatch FP8_PER_CHANNEL_PER_TOKEN` | `ModelOptFp8PcPtLinearMethod` exists but is unreachable through `MIXED_PRECISION`; the algo resolves, matches no branch, falls through to `UnquantizedLinearMethod`, and the checkpoint then fails to load with *no module or parameter named ...weight_scale* | hit directly; FP8_PB_WO had the same omission and was fixed upstream since our wheel |
| `[Model] Pass quant_config through GatedResidual` | `quant_config=None` hardcoded on all three projections, so hyper-connections are always unquantized. The docstring claims the opposite | 25% of decode GPU time on GB10 sits in these layers' BF16 GEMMs, on `cutlass_80_wmma_tensorop_bf16` |
| `[Model] Pass quant_config to ParallelLMHead` | omitted at both sites; a quantized head loads unquantized | +11% single-stream on GB10, no measurable quality cost |

Deliberately **not** in the branch:

- **`_is_sm103()` widened to sm_121.** We measured it: 36.45 stock vs 35.92 with the kernel
  provably dispatching, n=6 each from empty compile caches — not significant. Proposing it would
  be proposing a change with no demonstrated benefit. Kept as
  `patches/skinny-gemm-sm121-gate.patch` with the four blockers written up in
  `skinny-gemm-on-sm121.md`.
- **QSA fused multi-step draft decode.** Correct but worth nothing (`patches/qsa-fused-draft-decode.patch`).

## Push is blocked, and not by us

```
! [remote rejected] gb10-sm121-fixes -> gb10-sm121-fixes
  (refusing to allow an OAuth App to create or update workflow
   `.github/workflows/pre-commit.yml` without `workflow` scope)
```

The branch carries `.github/workflows/` from the PR head; our token has
`admin:public_key, gist, read:org, repo` and no `workflow`. To push:

```
gh auth refresh -h github.com -s workflow     # interactive, needs a browser
cd ~/vllm-fork && git push -u origin gb10-sm121-fixes
```

Until then the series in `patches/series-pr53896/` is the portable form — it applies to the PR
head with `git am --3way` and needs no fork access.
