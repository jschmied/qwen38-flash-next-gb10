# The 128-divisibility that blocks a layer class is a constant, not a kernel limit

`ModelOptFp8PbWoLinearMethod` refuses any layer whose dimensions are not both divisible by 128:

```python
_WEIGHT_BLOCK_SIZE: tuple[int, int] = (128, 128)     # modelopt.py:629
...
if output_size_per_partition % block_n != 0: raise ValueError(...)
if input_size_per_partition  % block_k != 0: raise ValueError(...)
```

We assumed that reflected the kernel. It does not.

## The kernel is general

`TritonFp8BlockScaledMMKernel` declares no shape constraint whatsoever:

```python
@classmethod
def is_supported(cls, compute_capability=None):
    if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        return False, "only CUDA-alike and XPU devices are supported."
    return True, None
```

and hands the block size through as a **runtime list** into a Triton kernel whose `group_n` and
`group_k` are runtime arguments (`offs_bsn = offs_bn // group_n`). There is no `can_implement`
override on that class. `CutlassFp8BlockScaledMMKernel.can_implement` checks only the *activation*
group shape `(1, 128)` — it never validates the weight block either.

So a single class constant makes an entire layer class unquantizable, on **every** GPU, not just
sm_121. That is worth reporting upstream regardless of what it buys us.

## Blocks need not be square

Flash-Next's hyper-connections are `(320, 10240)` and `(10240, 320)`. The natural blocks are
therefore **(64, 128)** and **(128, 64)** — transposed, because the tensors are. Nothing in the
Triton path cares; `group_n` and `group_k` are independent. A square-block assumption would have
silently handled half the tensors and skipped the rest, which is exactly what our first attempt did.

Our patch makes the block per-layer, read from a `block_size` field in the checkpoint's
`quantized_layers` entry, defaulting to the old constant.

## The blocker we could not remove: fusion

The down-projection still cannot be quantized, and the reason is neither the constant nor the
kernel. vLLM builds it as

```python
MergedColumnParallelLinear(
    hyper_hidden_size,
    [lora_rank, hc_count] + ([pad_size] if pad_size else []),   # [320, 4, 12] = 336
    ...)
```

when `use_combine=True`. **A blockwise scale cannot span a fusion boundary** — for the scale to
mean anything the block must divide every constituent shard, and one shard is `hc_count = 4` wide.
No useful block size divides 4, 320 and 12 together.

This is worth stating precisely because we spent a day reasoning about `(320, 10240)` read from the
checkpoint, while the layer vLLM actually instantiates is **336** wide. The error message
(`requires out_features divisible by 128, got 336`) is the first place that number appears
anywhere.

**Checkpoint shapes are not runtime shapes.** The same class of mistake as reading
`hf_quant_config.json` when vLLM reads `config.json`: the artefact you inspect is not the artefact
the code consumes.

## What that leaves

`input_mix_weight_up` is a plain `ReplicatedLinear`, unfused, `(10240, 320)` — quantizable at
(128, 64). That is 97 tensors and **0.66 GB/token**, half the hyper-connection total.

## The plumbing works. The result is still negative.

The per-layer block size loads, and vLLM's kernel selection adapts by itself — the log shows
**both** kernels chosen in one model:

```
Selected CutlassFp8BlockScaledMMKernel for ModelOptFp8PbWoLinearMethod   <- the 128x128 layers
Selected TritonFp8BlockScaledMMKernel  for ModelOptFp8PbWoLinearMethod   <- the (128, 64) layers
```

Cutlass declines the non-square block and Triton takes it, automatically. That confirms the whole
argument: the general kernel was always there, and only the constant stood in front of it.

And it is **still not worth shipping**:

| c | `lm_head` build | + hyper-connection `_up` | |
|---:|---:|---:|---|
| 1 | **36.3** | 35.0 | **−3.6%** |
| 2 | 52.3 | 55.5 | +6.1% |
| 4 | 73.0 | 76.1 | +4.2% |
| 8 | **115.8** | 113.8 | −1.7% |
| NLL/token | **0.9628** | 0.9713 | +0.88% worse |

**Slower at c=1 while removing 0.66 GB/token.** The kernel selection explains it: those layers went
from a BF16 cuBLAS GEMV to a Triton FP8 blockwise GEMM. We removed bytes and simultaneously moved
to a kernel that is slower for `(10240, 320)` at M=1 — cuBLAS has had a long time to optimise
skinny BF16 GEMV, and the Triton blockwise path has not.

**Removing bytes is not sufficient. You also have to land on a kernel that is at least as good.**
This is the third measurement in a row where the roofline over-predicted, and the third distinct
reason:

| lever | predicted | measured | why it missed |
|---|---|---|---|
| `shared_expert` | ~+8% | +1.9% | small matmul, latency-bound not bandwidth-bound |
| hyper-connection `_up` | ~+13% | **−3.6%** | forced onto a slower kernel |
| `lm_head` | ~+10% | +11% | large matrix, genuinely bandwidth-bound — the model held |

The roofline holds where the matrices are large and the kernel does not change. Both caveats matter
and neither is visible in the arithmetic.

## What is worth sending upstream anyway

The constant is still wrong, independent of our result. `_WEIGHT_BLOCK_SIZE = (128, 128)` blocks
layers that the Triton kernel handles fine, on every GPU. Whether a given layer *benefits* is a
separate question that can only be answered by measuring — which is impossible while the constant
refuses to let anyone try.
