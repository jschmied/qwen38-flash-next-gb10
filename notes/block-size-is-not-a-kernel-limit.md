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
