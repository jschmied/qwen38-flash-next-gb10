DRAFT — needs the user's go. TensorFold PR #242 reply to the maintainer (2026-10-03).
Thanks! Level is what I'd expect on both of those. On 0.6.3 only `Fp8BlockLinear.prefill` (128×128 block FP8) sends prompt rows through the lane matmul that #242 changed. `Fp4Linear`, `Fp8Linear` and `Mx8Linear` prefill through `_prompt`. local-inference-lab's Flash Next NVFP4 has MXFP8 dense layers, and the 27B NVFP4 has FP8 + NVFP4, so neither takes the changed path.

Our setup is a local Flash Next build with NVFP4 routed experts and 128×128 block-FP8 dense projections. On v0.6.3 itself, in-process prefill, `TF_QMMF_FUSED_ROWS=1000000000` (off) vs the default, two rounds, GB10:

| prompt | off | on |
|---|---|---|
| 8K | 6.01 / 6.04 s | 5.80 / 5.76 s (−3.4 / −4.6 %) |
| 32K | 24.07 / 24.19 s | 23.25 / 23.12 s (−3.4 / −4.4 %) |

The first 16 tokens were identical. So it pays only for checkpoints whose dense layers are block FP8.

Written with AI assistance (Claude Code).
