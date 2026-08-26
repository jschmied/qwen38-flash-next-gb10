# Who else is running this, and how

As of 2026-08-26, the day of release. None of these is a single-box vLLM deployment,
which is the only reason this repo exists.

| repo | stack | hardware | result |
|---|---|---|---|
| [MiaAI-Lab/…-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | SGLang TP2, NVFP4 | **2×** Spark, ConnectX-7 200 Gb RoCEv2 | **64 tok/s** single-stream, 117 aggregate @ c=2 (NEXTN 3/1/4) |
| [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp, mmap tensor pinning | 1× Spark | ~22 tok/s, `--parallel 1` |
| [lendome/llama.cpp-qwen4exp](https://github.com/lendome/llama.cpp-qwen4exp) | llama.cpp + PR #27742 | — | build recipe |
| [hocestnonsatis/qwen3.8-flash-next-16gb](https://github.com/hocestnonsatis/qwen3.8-flash-next-16gb) | llama.cpp UD-IQ1_S | 16 GB VRAM | extreme-quant route |

## What the dual-Spark deployment shows

Same checkpoint this repo uses (`RadixArk/Qwen3.8-Flash-Next-NVFP4`). Per node:

    GPU weights            ~62.5 GB
    pinned host PLE table  ~11   GB
    KV cache (956,800 tok)   9.0 GB
    free                   ~17.5 GB

Two things follow. **TP2 halves the weights**, which is how they sidestep the residency
problem rather than solve it — not a technique available on one box. And their pinned PLE
is ~11 GB per node, ~22 GB total, well under the 47.7 GiB table; whether that is sharding,
partial pinning, or a staging buffer is worth reading their `start.sh` for before designing
anything here.

Their KV figure is a useful reassurance either way: 956,800 tokens in 9.0 GB, because the
model has 2 KV heads and mostly linear attention.

## The SM121 risk, and why vLLM appears to dodge it

MiaAI had to patch SGLang so Qwen Sparse Attention falls back to Triton FlashDecoding when
`is_sm100_supported()` is false. vLLM PR #53896 ships that by construction:

```
nvidia/qsa.py:3    """NVIDIA QSA owner with Triton kernels."""
nvidia/qsa.py:74   return "QWEN4_EXP_QSA_TRITON"

nvidia/ops/qsa.py:791-792
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
```

QSA is Triton from the start, and the faster path is explicitly excluded for the 120 family —
which is sm_121. This was the largest unknown for a first boot here and it looks addressed
deliberately, not by accident. `low_latency_gemm.py:83` similarly gates a decode GEMM on
device capability (10, 3), so that path is simply not taken on GB10.

**Unverified.** Nothing here has been booted yet; this is code reading, not a result.
