DRAFT — needs the user's go. ashhart/TensorFold PR from jschmied/TensorFold:pr-mtp-fp8 into main (2026-10-01).

TITLE: Flash Next CUDA: read FP8 experts in the MTP drafter

A ModelOpt Flash Next export whose MTP layer has FP8 experts (per expert: e4m3 with a scale a tensor or a row, or
128×128 blocks) is refused at the check today, and if it got past the check, the loader would read those experts as
NVFP4 (`moe_nvfp4` takes any per-expert MoE as FP4 and asks for `weight_scale_2`).

The drafter's experts are already re-quantized to NVFP4 at load when they arrive as bf16 (`moe4_from_bf16`), and
drafts only propose tokens the target verifies. So FP8 drafter experts are dequantized to bf16 and take that same
path; nothing the target computes changes. FP8 in the main layers' experts stays refused, with a message: re-quantizing
those would change replies. The check accepts `FP8` on MTP expert layers only.

On one DGX Spark, our export with per-tensor FP8 MTP experts (block-FP8 dense layers, NVFP4 routed experts), refused
on main, now loads; three greedy replies are identical to the same model with NVFP4 MTP experts, and the drafts were
accepted 86 of 93 times (86 of 99 with the NVFP4 drafter). The block layout (`weight_scale_inv`, as in
`nvidia/Qwen3.8-Flash-Next-NVFP4`'s MTP experts) goes through the existing block-FP8 dequantizer; I have not loaded
that checkpoint.

## Tests

- `tests/cuda/nvfp4_tiny.py` gains `mtp_experts`: `"fp8"` writes per-expert e4m3 MTP experts with a scale a tensor,
  `"fp8_dequant"` writes stacked bf16 experts holding exactly those values times their scales (the same draws).
- `tests/cuda/test_flashnext_nvfp4_loader.py`: the FP8 drafter drafts and accepts exactly as the bf16 one with the
  same values (tokens, rounds, drafted, accepted, keeps, widths), and its drafts keep the serial tokens.
- `tests/test_hub_and_checks.py`: FP8 on `mtp.layers.0.mlp.experts` passes the check; FP8 on a main layer's experts
  is refused.

On one GB10: `tests/cuda/test_flashnext_nvfp4_loader.py` 26 passed (25 on main plus the new test).

Written with AI assistance (Claude Code); the author reviewed every change.
