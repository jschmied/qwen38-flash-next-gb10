# Choosing a quantization scheme for a layer vLLM won't quantize

Three schemes are available for a dense layer in a ModelOpt `MIXED_PRECISION` checkpoint, and the
choice is constrained more by **plumbing** than by quality. This is the reasoning we used for the
hyper-connections; it generalises.

## The constraints, in the order they eliminate options

**1. Shape.** `ModelOptFp8PbWoLinearMethod` — the blockwise-FP8 scheme the rest of the checkpoint
uses — requires **both** dimensions divisible by 128. The hyper-connections are `(320, 10240)` and
`320 % 128 = 64`, so this scheme cannot express them at all. Check first; it is a one-line test that
rules out the obvious choice.

| scheme | group | 320 divisible? |
|---|---|---|
| FP8_PB_WO | 128×128 | **no** |
| MXFP8 | 32 (along K) | yes |
| NVFP4 | 16 | yes |
| FP8 per-tensor | — | yes |

**2. Whether the method demands an activation scale.** `ModelOptFp8LinearMethod` registers
`input_scale` **unconditionally** when the checkpoint is FP8-serialized, initialises it to the
`float32.min` sentinel, and builds its kernel with `kFp8StaticTensorSym` for activations as well as
weights — i.e. **static W8A8**. Supply no calibrated activation scale and the kernel runs against
the sentinel: no error, plausible-looking output, wrong. This is the same class as the documented
`config_groups`/`quantized_layers` trap.

`ModelOptMxFp8LinearMethod` registers only `weight` and `weight_scale`. No activation scale, no
calibration, nothing to leave uninitialised.

**3. Whether the config gate passes.** MXFP8's `create_weights` raises unless
`is_checkpoint_mxfp8_serialized`. Under `MIXED_PRECISION`, `ModelOptMixedPrecisionConfig` constructs
`mxfp8_config` with that flag hardcoded `True`, so the gate is free. Verify rather than assume — the
equivalent flag is *not* free for every sub-config.

## Only then, quality — and here it did not discriminate

Mean relative error against BF16 on six hyper-connection tensors:

| scheme | error |
|---|---|
| per-tensor FP8 | 2.251% |
| per-output-channel FP8 | 2.248% |
| MX group-32 | 2.252% |

Within 0.06 pp of each other, and equal to the ~2.25% that lovedheart's dense projections show.
**That number is the FP8 E4M3 representation floor, not a property of granularity** — these weights
have no outliers, so finer scaling buys nothing. When quality does not discriminate, pick on
plumbing risk, which is what sent us to MXFP8.

Measure this before choosing. Had the tensors carried outliers, per-tensor would have been much
worse and the argument would have run the other way.

## Do not hand-roll the encoding

MXFP8 scales are E8M0: `ceil(log2(amax / fp8_max)) + 127`, clamped to `[0, 254]`, stored as `uint8`,
dequantised as `exp2(byte - 127)`. Every part of that is easy to get subtly wrong, and a wrong bias
produces fluent garbage rather than an error.

vLLM ships the quantizer — `vllm.model_executor.layers.quantization.utils.mxfp8_utils`:

```python
from vllm...mxfp8_utils import _mxfp8_e4m3_quantize_torch, dequant_mxfp8_to_bf16
q, scale = _mxfp8_e4m3_quantize_torch(w.float().cuda())
rec = dequant_mxfp8_to_bf16(q, scale)          # round-trip check, same code path
```

Use it. The bytes are then produced by the same code that will decode them, which removes the
entire class of encoding error. The same argument applied to FP8_PB_WO, where we instead determined
the convention empirically because no such helper was exposed — `w_fp8 * scale` reconstructs at
2.2489%, `w_fp8 / scale` at 5.7e8%.
