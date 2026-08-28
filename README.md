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

**Status: working, and it scales.** `RadixArk/Qwen3.8-Flash-Next-NVFP4` serves coherently on
vLLM on one GB10 — 76.6 GiB resident with the PLE offloaded to host, **17.1 tok/s** single-stream
and **266.8 tok/s aggregate at 48 concurrent streams** (TTFT 1.6 s, no queueing). Full numbers in
[notes/results-radixark-vllm.md](notes/results-radixark-vllm.md); the load trace behind the
concurrency figures is in [notes/load-and-waits.md](notes/load-and-waits.md).

**The PLE offload is not the bottleneck, and swap gets *cheaper* under load.** Traced with
`/proc` + `/metrics` counters (nothing instrumented): the offload worker never exceeds 24% of one
core, and major faults per token **fall 4.4x** from c=1 to c=48 — batched tokens share n-gram rows,
so the marginal token is far cheaper than the first. That is an argument for running this model at
concurrency rather than a caution against it. Every wait we could observe is a queueing wait
governed by `--max-num-seqs`; the value we originally shipped (2) costs **4x aggregate throughput**
at c=8.

Two things are needed, and both are contributions this repo can make to the field:

1. **A one-line gate change** so the FP8 PLE is accepted on a ModelOpt/NVFP4 body. This corrects
   published checkpoint tables that list RadixArk as not loading on vLLM — it does.
2. **`--cap-add=SYS_PTRACE`** when running vLLM's official `VLLM_PLE_CPU_OFFLOAD` in Docker.
   `rebuild_cuda_tensor` needs `pidfd_getfd`; default seccomp denies it, and the engine dies
   *after* both workers load all 206 shards with only
   `Engine core initialization failed. Failed core proc(s): {}`. Undocumented as far as we can
   tell, and it will hit anyone taking the official offload path in a container.

**Honest positioning:** on **single-stream** decode we are behind the field — 17.1 tok/s with no
speculation, against 22 ([0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark),
llama.cpp), 27 ([Death-By-Tokens](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark),
SGLang + HashK-PLE), 28.2 ([dolf3131](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark),
vLLM + MTP k=2) and 31–50 ([paragontasx](https://github.com/paragontasx/qwen38-flash-next-dgx-spark),
llama.cpp). Speculative decoding is the untried lever here and would close much of that gap.

On **aggregate** throughput the picture inverts: 266.8 tok/s at 48 streams, roughly an order of
magnitude above any published single-stream figure. We have not seen anyone else measure it, and
llama.cpp builds cannot (`--parallel 1`). Which number matters depends entirely on whether the box
serves one caller or several.

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

- **[Failure modes](notes/failure-modes.md) — read this first if anything is wrong.** Every
  failure hit here, organised by what you observe, with the signature that distinguishes causes
  that present identically. Four different things produce "it loads but the output is wrong".
- **[The full working result](notes/results-radixark-vllm.md)** — 76.6 GiB resident, 17.3 tok/s
  single-stream, 32.4 tok/s at two concurrent streams.
- **A one-line gate makes RadixArk NVFP4 load**, contradicting checkpoint tables that list it as
  incompatible with vLLM. Its PLE is already in the exact FP8 format vLLM implements; only the
  `isinstance(quant_config, Fp8Config)` check rejects it, because the *body* is NVFP4.
- **`--cap-add=SYS_PTRACE` is required** for vLLM's official `VLLM_PLE_CPU_OFFLOAD` inside
  Docker. `rebuild_cuda_tensor` needs `pidfd_getfd`; default seccomp denies it, and the engine
  dies ten minutes in — after all 206 shards load — with only `Failed core proc(s): {}`.
- **[No source build is required](notes/why-no-source-build.md).** vLLM PR #53896 changes
  one CUDA kernel signature, and Qwen4 needs the new `sigmoid` value — but that op is
  reached only under speculative decoding. Run without speculation and the stock
  prebuilt kernel is never called. This turns a multi-hour risky build into a file overlay.
- **CPU offload does work here — an earlier claim on this page was wrong.** We had reasoned that
  because GB10 shares one 128 GB pool between CPU and GPU, "offload to host" frees nothing. In
  practice the served model reports **76.61 GiB** device-consumed with the PLE held by the
  offload worker, leaving 30.99 GiB for KV. Caveat we are explicit about: this box had a 64 GiB
  swapfile active during the run, and **we did not measure how much of the PLE was resident
  versus paged out**. Treat the swapfile as part of the recipe until someone measures it.
- **The n-gram table is a lookup, not a GEMM**, and is pre-split into 128 shards across
  10 files — so file-backed `mmap` is a viable mechanism too, letting the kernel
  hold hot rows and evict the rest.
- **Compressing the table beats offloading it.**
  [Death-By-Tokens](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark)
  re-hashes the PLE trainlessly from 51 GB to **12.8 GB** in about six minutes, and spends the
  freed ~16 GB on an MTP draft head that roughly doubles decode. Reconstruction cosine is only
  ~0.50, yet their code benchmark went *up* (12/12 vs 10/12) because the model's own PLE
  conv+gating filters the retrieved values. That is a better answer to the problem this repo
  exists to solve, and it is theirs.

## Layout

    scripts/serve-flashnext.sh   first-boot serve config (no speculative decoding)
    scripts/apply-pr53896.sh     overlay the PR's Python files onto a venv
    notes/failure-modes.md       everything that went wrong, by symptom  <- start here
    notes/load-and-waits.md      where time goes under concurrency (PLE is not the bottleneck)
    notes/single-stream-limit.md what limits n=1 (BF16 GEMV on unquantized dense weights)
    notes/fetching-a-slice.md    diffing lfs.sha256 to download 12 GiB instead of 123
    notes/fp8-mixed-checkpoint.md the +71% checkpoint switch, and why MTP stops paying
    notes/quantizing-lm-head.md   +11% free, and the three blockers that stopped everyone
    notes/choosing-a-quant-scheme.md how to pick a scheme when plumbing, not quality, decides
    notes/ple-access-pattern.md   why the biggest object in the checkpoint is a small cost
    notes/quantizing-shared-expert.md a lever that measures worse than it models -- and why
    notes/block-size-is-not-a-kernel-limit.md a constant, not a kernel, blocks a whole layer class
    notes/results-radixark-vllm.md  the working config and its measurements
    notes/the-field.md           who else is running this, and how
    notes/log.md                 running record, including the dead ends

## Related upstream work

| | |
|---|---|
| #53896 | `[Model] Support Qwen3.8-Flash-Next` — 111 files, unreviewed |
| #53899 | PLE-Offload (to host memory) |
| #53908 | auxiliary-GPU offloading for the N-gram / PLE table |
| #53909 | `Add qwen4 fuse op` |

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.
