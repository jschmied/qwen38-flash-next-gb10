POSTED 2026-10-01 as https://github.com/ashhart/TensorFold/pull/176 — user's go 2026-10-01 ("post whats ready"). ashhart/TensorFold PR from jschmied/TensorFold:pr-loader-guards.

TITLE: Flash Next CUDA: refuse quantized bytes where the loader reads bf16 values

Two places in the Flash Next NVFP4/ModelOpt loader turn a weight into bf16 by casting, whatever it holds:

- [`weight_bf16`](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/families/qwen4_exp/cuda/weights.py#L190-L196) dequantizes block FP8 and casts
  everything else (I wrote that cast in #126). An MXFP8 or per-tensor FP8 `lm_head` would load as raw e4m3 values.
- [`dense()`](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/families/qwen4_exp/cuda/weights.py#L78-L97) returns a weight without a scale as a bf16
  linear, so packed NVFP4 bytes (`uint8`) on a non-expert layer would load as numbers.

Either way the shapes are right, nothing fails, and the model decodes fluent garbage. We hit exactly that on vLLM
with FP8_PB_WO layers read as BF16, and it took a while to see, so a loud refusal is worth a few lines. The checkpoint check
also accepts `NVFP4`/`W4A16_NVFP4` on any layer, and published exports do put it elsewhere:
`myllmbox/Qwen3.8-Flash-Next-hibrid48` lists 436 `W4A16_NVFP4` layers, among them the DeltaNet projections and the
head.

## The change

- `_plain(name, w)` in `cuda/weights.py`: a weight a bf16 linear takes must be bf16, fp16 or fp32, else a
  `ValueError` naming the layer and its dtype. `weight_bf16` and `dense()`'s no-scale branch go through it.
- `qwen4_exp.check`: `NVFP4` / `W4A16_NVFP4` outside the routed experts is refused before any weight is read, with
  the first such layer named.

Every supported checkpoint passes: the gate only refuses NVFP4 on layers whose name has no `experts` part, and ours
(`MIXED_PRECISION`: NVFP4 `...mlp.experts`, FP8_PB_WO elsewhere) and the published NVFP4 configs list NVFP4 on
`mlp.experts` only.

## Tests

`tests/test_flashnext_plain_weights.py` (bf16/fp16/fp32 pass, uint8/int8/e4m3 refused) and a case in
`tests/test_hub_and_checks.py` (W4A16 NVFP4 on `linear_attn.in_proj_qkv` refused by name). CPU suite: the same
failures as main on this machine (missing MLX and similar), none new.

Written with AI assistance (Claude Code); the author reviewed every change.
