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

**Status: working.** `RadixArk/Qwen3.8-Flash-Next-NVFP4` serves coherently on vLLM on one
GB10 — 76.6 GiB resident with the PLE offloaded to host, **17.3 tok/s** single-stream and
**32.4 tok/s** across two concurrent streams (6% per-stream loss). Correct on every spot-check.
Full numbers and the two required changes in
[notes/results-radixark-vllm.md](notes/results-radixark-vllm.md).

Two things are needed, and both are contributions this repo can make to the field:

1. **A one-line gate change** so the FP8 PLE is accepted on a ModelOpt/NVFP4 body. This corrects
   published checkpoint tables that list RadixArk as not loading on vLLM — it does.
2. **`--cap-add=SYS_PTRACE`** when running vLLM's official `VLLM_PLE_CPU_OFFLOAD` in Docker.
   `rebuild_cuda_tensor` needs `pidfd_getfd`; default seccomp denies it, and the engine dies
   *after* both workers load all 206 shards with only
   `Engine core initialization failed. Failed core proc(s): {}`. Undocumented as far as we can
   tell, and it will hit anyone taking the official offload path in a container.

**Honest positioning:** 17 tok/s single-stream is slower than the field's best on this hardware —
[paragontasx](https://github.com/paragontasx/qwen38-flash-next-dgx-spark) reports 31–50 tok/s on
llama.cpp, [Death-By-Tokens](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark)
~27 tok/s on SGLang, both with speculative decoding and 262K context against our 8K. What this
path adds is **concurrency**, which llama.cpp cannot do (`--parallel 1`). Speculative decoding is
the obvious next lever and is not yet enabled here.

### What cost a day getting here, and the rule that would have prevented it

The output was fluent garbage until the very end. The cause was **two of my own 206 files:
size-correct and byte-corrupt** — `model-bf16-00011.safetensors` (dense BF16 body) and
`model-plefp8-00000.safetensors` (PLE shards 0-12), the two still being written when my download
stalled. I compared **sizes** against the HF API, saw 418/418 agree, deleted aria2's `.aria2`
control files, and called it verified. `lfs.sha256` was in the same API response.

**Verify `lfs.sha256`, not file size.** A size-correct corrupt shard loads without error, reports
sane shapes and dtypes, produces correct-magnitude activations, and yields *fluent* token salad.
Because it is a property of the weights, the garbage is **invariant to every configuration change
you can think of** — which reads exactly like an environmental or kernel fault. I eliminated
twenty-odd hypotheses that way and each clean elimination made the wrong conclusion look better
supported. It also survives naive content checks: I validated the PLE against the official BF16
table at cosine 0.999635, sampling row 0 — inside the intact head of the corrupt file.

```bash
curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true&blobs=true" \
  | jq -r '.[]|select(.lfs)|"\(.lfs.sha256)  \(.path)"' > SHA256SUMS
sha256sum -c SHA256SUMS
```

Final tally: **204 of 206 clean, 2 corrupt.** Two upstream issues opened on the strength of the
wrong conclusion have been retracted
([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
[dolf3131#1](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark/issues/1)).

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
