DRAFT — needs the user's go. vllm-project/vllm PR #56757 (2026-09-20).

Asking whether the disk backend should be generalized behind `EngramConfig` to `qwen4_exp`, because
on unified memory the merged pinned path is not a slower option — it is no option.

**Box**: NVIDIA DGX Spark, GB10, sm_121, aarch64, TP1, 121.6 GiB unified (`MemTotal`), 64 GiB swap.
vLLM `0.28.1rc1.dev524+g5db652225` plus the #53899 offload worker. Checkpoint: a locally derived
FP8-head variant of `RadixArk/Qwen3.8-Flash-Next-NVFP4` — 122.9 GiB on disk, of which the PLE table
is **47.7 GiB** in 10 FP8 shards.

### The weights alone exceed host memory

| | |
|---|---:|
| non-PLE weights | 75.2 GiB |
| PLE n-gram table (FP8) | 47.7 GiB |
| **total, before any KV cache or CUDA context** | **122.9 GiB** |
| `MemTotal` | 121.6 GiB |

Host and device are the same physical pool here, so "offload to host memory" moves the allocation
between accounting buckets without creating capacity. What makes the model serve is not where the
table is but that its pages are **evictable**: the table is cold during load, the kernel pushes it
out as the main worker claims RAM, and it stays out.

Measured on this box, serving now:

| phase | `MemAvailable` | swap used |
|---|---:|---:|
| load, minimum | 5.5 GiB | — |
| load, after the table is paged out | 38.7 GiB | 51.4 GiB peak |
| steady state, serving (152 samples) | 5.3 GiB | **49.8 GiB** |

`MemAvailable` *recovering* mid-load is the mechanism working, which reads as the opposite from a
memory graph. Engine figures for the same run: `GPU KV cache size: 869,444 tokens`,
`Free memory on device (114.02/121.63 GiB)`. Decode 24.9 / 24.5 / 21.6 tok/s at c=1, 300 tokens,
no speculative decoding.

### Why #54371's merged backends do not reach this state

[`ngram_embedding.py`](https://github.com/vllm-project/vllm/blob/3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L703-L707)
selects between exactly two:

- [`Qwen4ExpPLEDeviceEmbedding`](https://github.com/vllm-project/vllm/blob/3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L325) — 122.9 GiB resident on device, over the pool before KV.
- [`Qwen4ExpPLEPinnedHostEmbedding`](https://github.com/vllm-project/vllm/blob/3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L387) — [`allocate_embedding_weight`](https://github.com/vllm-project/vllm/blob/3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L430-L443) is unconditionally `pin_memory=True` for the complete table. Pinned pages cannot be evicted, so the 47.7 GiB stays resident and the 122.9 GiB above has nowhere to go.

`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` is not an escape — it gates `model_executor/offloader/uva.py`
and `platforms/xpu.py`, not this path. ETP sharding divides the table across ranks; at TP1/DP1 there
is one rank.

**Not measured, deliberately**: I have not run the pinned backend on this box. Pinning a table of
this size against this pool has hard-reset the machine before, and the arithmetic above does not
need a confirming crash. If a maintainer wants the failure signature recorded I will run it, but I
would rather not volunteer the box for a result that is already implied.

### The ask

This PR's framing already contains the reason it matters here — *"a mapped file cannot be read
through UVA, which needs page-locked memory, and pinning the mapping would return the table to
RAM"* — and your B200 TP4 numbers have disk at 14,168 vs 13,632 tok/s, so a mapped backend is not
asking for a throughput trade.

Since `EngramConfig` is shared between DeepSeek-V4.1 Engram and Qwen4Exp PLE, and `qwen4_exp`'s
[`Qwen4ExpPLEEmbedding`](https://github.com/vllm-project/vllm/blob/3116c5d06bfe76501b3dd6b5434bfc7f3274f5e7/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L52-L96)
is an ABC whose storage contract is the single method `allocate_embedding_weight`: **would you
prefer `disk_offload_dir` to be generalized behind `EngramConfig` so a `qwen4_exp` backend is one
subclass, or kept DeepSeek-specific with `qwen4_exp` carrying its own?** I am happy to write and
test the `qwen4_exp` side on GB10 either way — as an additive third backend, leaving pinned the
default, so nothing changes for the multi-GPU deployments #54371 was built for.

Two related attempts exist and both predate #54371, so neither applies to current main: #54129
(mmap from the checkpoint's own safetensors, no worker) and #54070 (disk + page cache, built on
#53899's worker, which main no longer has). I have not contributed to either.

Unrelated to the above, and complementary to @yunwei37's read-path note from this morning: our
reader work on a different model found the same shape, that at low concurrency the orchestration
around a cached read dominates the read.

_AI assistance (Claude Code) was used for this analysis; I reviewed every number against the run
that produced it._
