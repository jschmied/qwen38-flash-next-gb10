# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Can Qwen's Qwen4-architecture preview be made to run on a single GB10 with 128 GB of
unified memory? This repo is the working record of finding out — including the parts
that do not work.

**Status: in progress.** Weights downloading; the model has not yet been booted.

## The problem in one table

| build | weights |
|---|---|
| `Qwen/Qwen3.8-Flash-Next` (BF16) | 335.3 GiB |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | **125.9 GiB** |
| **usable on this box** | **~117 GiB** |

The NVFP4 build is the only one in range, and it is still ~9 GiB too large. It splits
cleanly, which is what makes the attempt worth making:

| component | files | size |
|---|---|---|
| main model (experts NVFP4 + some BF16) | 196 | **78.2 GiB** |
| PLE / n-gram table, already FP8 | 10 × `model-plefp8-*` | **47.7 GiB** |

The main model alone fits comfortably. Everything hinges on keeping the n-gram table
out of resident memory.

## What the model is

`Qwen4ExpForConditionalGeneration` / `qwen4_exp_text`:

- 48 layers, hidden 2560, **512 experts / 10 active**, `moe_intermediate 640`
- 24 Q heads / 2 KV heads, head_dim 256, **262144** native context
- `layer_types`: 3 × `linear_attention` + 1 × `full_attention`, repeating (GDN hybrid)
- n-gram: `ngram_size 3`, `ngram_vocab_size_base 20_000_000`, `split_ngram_parts 128`
  → 20e6 × 2560 = **51.2B** parameters, attached at layer 1 as
  `layers.1.ple.ple_embedding.ngram_embedding.shard_0…127`
- `output_gate_type = sigmoid` — this one matters, see below

## Findings so far

- **[No source build is required](notes/why-no-source-build.md).** vLLM PR #53896 changes
  one CUDA kernel signature, and Qwen4 needs the new `sigmoid` value — but that op is
  reached only under speculative decoding. Run without speculation and the stock
  prebuilt kernel is never called. This turns a multi-hour risky build into a file overlay.
- **CPU offload is useless here.** GB10 shares one 128 GB pool between CPU and GPU, so
  "offload to host" frees exactly nothing. Upstream's two offload paths (#53899 to host
  memory, #53908 to a second GPU) both assume hardware this box does not have.
- **The n-gram table is a lookup, not a GEMM**, and is pre-split into 128 shards across
  10 files — so file-backed `mmap` is the mechanism that could work, letting the kernel
  hold hot rows and evict the rest. Nobody upstream is building that.

## Layout

    scripts/serve-flashnext.sh   first-boot serve config (no speculative decoding)
    scripts/apply-pr53896.sh     overlay the PR's Python files onto a venv
    notes/                       reasoning, measurements, dead ends

## Related upstream work

| | |
|---|---|
| #53896 | `[Model] Support Qwen3.8-Flash-Next` — 111 files, unreviewed |
| #53899 | PLE-Offload (to host memory) |
| #53908 | auxiliary-GPU offloading for the N-gram / PLE table |
| #53909 | `Add qwen4 fuse op` |

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.
