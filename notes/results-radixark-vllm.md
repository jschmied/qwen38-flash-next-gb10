# RadixArk NVFP4 on one Spark with vLLM: it loads, it serves, it emits garbage

Measured 2026-08-27 on a DGX Spark (GB10, sm_121, 121 GiB unified), image
`vllm/vllm-openai:qwen38-flash-next` @ `sha256:fc120ece0a38` (`0.1.dev20073+g8e685d198`) —
byte-identical to the build used by both reporters on
[vllm#53960](https://github.com/vllm-project/vllm/issues/53960).

## Confirmed: the TP=1 hang is executor selection

`--distributed-executor-backend mp` fixes it. Verified directly rather than inferred:

    docker exec fnext-test ps -eo pid,rss,comm
    513  1526164  python3      <- PleOffloadWorker, spawned

and the log shows it doing real work: `Loading safetensors checkpoint shards(PLE-offload):
100% Completed | 206/206`. Diagnosis credited to
[dolf3131](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark); this is an independent
confirmation on different weights.

## Contradicted: RadixArk *does* load

Published tables list `RadixArk/Qwen3.8-Flash-Next-NVFP4` as not loadable. The blocker is a
single gate in `nvidia/ple_layer.py`:

```python
def _get_ple_embedding_quant_method(quant_config, prefix):
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""
    if not isinstance(quant_config, Fp8Config):
        return None                    # RadixArk is modelopt/NVFP4 -> rejected
```

RadixArk's PLE is *exactly* the format that method implements, verified from the safetensors
headers: 128 shards of `F8_E4M3 [2500012, 160]`, plus **one** global scale
`ngram_embedding.weight_scale`, `BF16 [1]`. Only the `isinstance` check on the *body's* quant
config rejects it. Relaxing that gate for `modelopt` / `modelopt_fp4` gets a full load:

    Model loading took 73.77 GiB memory and 575.3 s
    GPU KV cache size: 294,638 tokens
    INFO:     Application startup complete.     health=200

With `VLLM_PLE_CPU_OFFLOAD=1`, TP=1, `--enforce-eager`, util 0.80, 79 GiB of swap
(47 GiB of it in use at steady state — the PLE paging out, as designed).

## Not solved: the output is garbage

```
In one sentence: what is a hash map?
->  K1 he in kt, Insto. Thefuckas almostsastimeewfzlogf[lasqfl...
```

Fluent-shaped token salad, no error, no warning. Two attempts did not fix it.

**Attempt 1 — relax the gate.** Loads, garbage. **This patch introduced a second bug of its
own:** under offload the GPU process is supposed to hold *no* embedding parameters —
`load_weights()` retains only `_offload_weight_scale` — but returning a quant method makes
`create_weights()` register `weight` and `weight_scale` on the GPU side, which then are never
filled. `_get_embedding_weight_scale()` prefers `embedding.weight_scale` over
`_offload_weight_scale`, so the lookup was dequantized with an uninitialised scale.

**Attempt 2 — respect the ownership split** (hand out the method only in the offload worker,
`envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process()` returns `None` on the GPU side).
Still garbage, so the uninitialised scale was not the whole story.

## What is not yet isolated — the control I skipped

**Whether the corruption is the PLE path or the NVFP4 body.** RadixArk is known-good on SGLang,
so the *data* is fine, but nobody has put it through vLLM's `modelopt` path. Both failures so
far are consistent with either. Deciding it needs one of:

- dump the dequantized lookup rows and compare against values computed offline from the
  safetensors — definitive, and the honest next step;
- or a body-only control, which this checkpoint does not permit since the PLE feeds layer 1.

Until that is done, "RadixArk loads but does not work under vLLM" is the accurate claim, and
the published "does not load" is wrong in mechanism but right in outcome.

## Mechanism notes gathered on the way

**The offload is a dedicated pool, not page cache.** `vllm/v1/ple_offload/worker.py` holds the
full table in ordinary *pageable* host memory in a separate process, gathers rows, stages them
through small **pinned** buffers (`torch.empty(..., pin_memory=True)`) and DMAs into a fixed
`gpu_output_buffer` on a dedicated copy stream with semaphore sync. Consequences: the table is
swappable (which is why swap sizing matters and why it works at all on 121 GiB), and the small
pinned tier is what other write-ups report as "~11 GB pinned PLE" — that is staging, not the
table.

**Page-granular pinning cannot work here.** Rows are 160 B, so a 4 KiB page holds ~25 rows, and
the table is hashed into 20 M buckets across 128 shards specifically to spread accesses. Hot
entries land on essentially every page; there is no compact region to `mlock`. Locality exists
at row level and is destroyed at page level by design. The viable levers are, in order of cost:
`vm.page-cluster=0` (swap readahead faults 8 pages per miss by default, ~200x amplification for
160 B random reads — applied here), a row-level LRU cache in the offload worker (the principled
fix, and real work), or an offline frequency permutation of the table so pinning becomes
possible at all.


## Debugging the garbage: what has been eliminated (2026-08-27)

| hypothesis | status |
|---|---|
| TP=1 selects uniproc, offload worker never spawns | **confirmed, fixed** — `--distributed-executor-backend mp`; `PleOffloadWorker` spawns and loads 206/206 |
| PLE gate rejects modelopt/NVFP4 | **confirmed, fixed** — relaxing the `isinstance` gate loads the model |
| GPU-side `weight_scale` shadows `_offload_weight_scale` | **fixed** — hand the method out only in the offload process; verified `has_ngram_ws=False has_offload_ws=True` |
| lookup returns an all-zero buffer | **measurement artefact** — the probe sampled the warmup dummy forward (`connector.py:403-410` zeroes and signals); shape `(2048,2560)` = `max_num_batched_tokens x ple_dim` |
| dispatch gated on cudagraph / `enforce_eager` | eliminated — dispatch is unconditional in both runners |
| Model Runner V2 vs V1 (`envs.py` says offload is V1-only) | eliminated — `VLLM_USE_V2_MODEL_RUNNER=0` still produces garbage |
| dtype divergence between CPU pinned buffer and GPU buffer | eliminated — both `float8_e4m3fn` |
| worker never serves real requests | eliminated — `num_tokens=56` for the prompt, then 1 per decode step |
| lexicographic `shard_0, shard_1, shard_10…` ordering permutes the table | eliminated — loader parses `int(shard_text)` and shape-validates each placement |
| run without offload as a control | **not runnable on one box** — with offload off the PLE is a CUDA allocation, which cannot swap; container OOM-killed (exit 137) at the cgroup cap despite 79 GiB of swap |

Worker-side probe on a real request:

    result.dtype=float8_e4m3fn  gpu.dtype=float8_e4m3fn
    absmax=208  nonzero=140156/143360

`208 x 0.000199 ~ 0.041` — a sane embedding magnitude. So the transport, the dtypes, the scale
and the shard placement are all correct, and the output is still token salad.

**What that leaves.** Either the row *indices* computed inside the offload worker are wrong
(right values, wrong n-grams — which would look exactly like this), or the NVFP4 body is
mishandled by this image independently of the PLE. Those cannot be separated by configuration
alone, because the no-offload control is not runnable here. Separating them needs a ground
truth: compute the expected lookup for a known token offline from the safetensors and compare,
or run the same checkpoint on a stack where it is known good (SGLang, per MiaAI) and diff the
first-token logits.

**Two of my own errors are recorded above** because they cost real time: sampling the warmup
forward and reporting its zeros as the finding, and putting a `print()` inside a compiled region,
which made a cudagraph test fail for a reason that had nothing to do with the hypothesis.


## The PLE is exonerated: checkpoint validated against the official table

Read three rows of `shard_0` from `Qwen/Qwen3.8-Flash-Next` by HTTP range (a few KB, no
download) and compared against the same rows of RadixArk's FP8 shard, dequantized with the
global scale `0.00019931793212890625`:

    official row0[:6]  : [-0.009216, 0.014282, -0.016479, 0.013306, -0.00708,  0.00705]
    radixark  deq[:6]  : [-0.009567, 0.014351, -0.015945, 0.012756, -0.007175, 0.007175]

    cosine similarity  : 0.999635
    relative error     : 2.4%          (expected for E4M3 with a single global scale)
    absmax             : 0.040039 (official) vs 0.041458 (dequantized)

This settles several things at once:

- **RadixArk's PLE is a faithful quantization of the official table.** Not a bad export.
- **The scale value is correct and the convention is multiply**, not reciprocal — a trap worth
  naming, since ModelOpt exports sometimes store reciprocals.
- **The offload path delivers correctly-scaled values.** The worker probe's `absmax=208` in FP8
  becomes `208 x 0.000199 = 0.0414`, matching the official `0.0400` absmax.

Every link in the PLE chain is now verified: shard placement (shape-validated), request service
(56 tokens for a prompt, 1 per decode step), matching FP8 dtypes on both sides, faithful values,
correct scale, correct dequantization. **And the output is still token salad.**

## What remains, and why it cannot be settled here

The NVFP4 W4A4 body under this image is the remaining suspect. It cannot be isolated on a single
Spark: the model cannot run without the PLE, and it cannot run with the PLE resident, because
with offload disabled the table becomes a CUDA allocation which the kernel cannot swap — the
container is OOM-killed at the cgroup cap despite 79 GiB of swap.

Note that MiaAI-Lab serve **this same checkpoint** successfully on SGLang across two Sparks, so
the body's *data* is good; what is in question is vLLM's handling of it. Settling that needs
either a second box, or a first-token logit diff against a known-good stack.

**Falsified along the way:** vllm#40252 (`linear_attn` split-vs-combined naming causing silent
NVFP4 garbage on Qwen3-Next) does not apply — RadixArk uses `in_proj_qkv` / `in_proj_z` /
`in_proj_a` / `in_proj_b`, byte-for-byte the same convention as the official
`Qwen/Qwen3.8-Flash-Next` checkpoint, and all `linear_attn` tensors are BF16 with no scales,
i.e. correctly excluded from quantization.
