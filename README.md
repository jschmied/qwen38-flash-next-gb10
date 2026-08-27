# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Can Qwen's Qwen4-architecture preview be made to run on a single GB10 with 128 GB of
unified memory? This repo is the working record of finding out — including the parts
that do not work.

> **Prior art, and the number to beat.** [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
> already runs this model on a Spark via **llama.cpp**, by pinning the n-gram tensor to the CPU
> backend and letting mmap serve it from NVMe (`-ot "per_layer_token_embd=CPU" -lm mmap`):
> 103.7 GiB on disk, ~76.9 GiB resident, **~22 tok/s decode**. That work came first and it works.
>
> **This repo takes the vLLM route instead**, for one reason: their setup is limited to
> `--parallel 1` (concurrent requests crash), and speculative decoding gave them no speedup.
> Concurrency and prefix caching are what vLLM would add. If that does not pan out, their
> answer is the better one and this repo will say so.

**Status: the garbage was my checkpoint, not the model, the recipe or the environment.**
Two of my 206 safetensors files were **size-correct and byte-corrupt** —
`model-bf16-00011.safetensors` (dense BF16 body weights) and `model-plefp8-00000.safetensors`
(PLE shards 0-12). They are exactly the two files still being written when my download stalled.
I compared file **sizes** against the HF API, saw 418/418 agree, deleted aria2's `.aria2` control
files, and called the download verified. HuggingFace publishes `lfs.sha256` in the same API
response I was already parsing, and I did not use it. Re-fetching now; this page will say plainly
whether it then serves.

Everything below that reads as a finding about the model, the quantization or the driver was
reasoning on top of corrupt weights, and the eliminations it reports are worth nothing. The two
upstream issues I opened on the strength of it have been retracted
([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
[dolf3131#1](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark/issues/1)).

### The one thing here worth another reader's time

**Verify `lfs.sha256`, not file size.** A size-correct corrupt shard is close to invisible: it
loads without error, reports sane tensor shapes and dtypes, produces activations of correct
magnitude, and yields *fluent* token salad. Because it is a property of the weights, the garbage
is **invariant to every configuration change you can think of** — which reads exactly like an
environmental or kernel fault and sends you hunting through cudagraph modes, executors, attention
backends and driver versions. I eliminated more than twenty hypotheses that way, and each clean
elimination made the wrong conclusion look better supported.

It also survives a naive content check. I validated the PLE against the official BF16 table and
got cosine **0.999635** — but I sampled row 0, which lived in the intact head of the corrupt
file, and took that as proof the checkpoint was sound.

```bash
# what I should have run, and now do (HF publishes the hashes; use them)
curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true&blobs=true" \
  | jq -r '.[]|select(.lfs)|"\(.lfs.sha256)  \(.path)"' > SHA256SUMS
sha256sum -c SHA256SUMS
```

Final tally on this checkpoint: **204 of 206 files clean, 2 corrupt.**

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
    notes/the-field.md      who else is running this, and how
    notes/                       reasoning, measurements, dead ends

## Related upstream work

| | |
|---|---|
| #53896 | `[Model] Support Qwen3.8-Flash-Next` — 111 files, unreviewed |
| #53899 | PLE-Offload (to host memory) |
| #53908 | auxiliary-GPU offloading for the N-gram / PLE table |
| #53909 | `Add qwen4 fuse op` |

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.
