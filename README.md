# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Can Qwen's Qwen4-architecture preview be made to run well on a single GB10 with 128 GB of
unified memory? This repo is the working record — including, deliberately, the parts that do not
work and the claims of our own we had to withdraw.

**Status: working, fast, and usable.** `17.1 → 36.5 tok/s` single-stream on one box, via three
measured levers. The model serves 262K-capable context, tool calls, and vision.

| lever | single-stream | note |
| --- | --- | --- |
| `RadixArk/…-NVFP4` as published | 17.1 | dense projections left in BF16 |
| + dense projections FP8 (lovedheart) | 23.7 | same size, +39% |
| + `lm_head` FP8 (ours, first on GPU for this model) | 26.1 | no measurable quality cost |
| + MTP k=2 | **36.5** | +35%; the head lever **doubles** under speculation |

Aggregate is a **different config and not comparable to the row above it**: the 266.8 tok/s at 48
streams in [load and waits](notes/load-and-waits.md) was measured on the *baseline* checkpoint
(its c=1 row is the 17.1 above), MTP off, with short prompts. On the shipped config with
4000-token inputs we measure **~100 tok/s at c=16 and 110 at c=32**, and c=32 costs 30.7 s TTFT.
Which number matters depends on whether the box serves one caller or several — see
[agentic speed is TTFT-bound](notes/speculation-on-flash-next.md).

## Start here

- **[REPRODUCE.md](REPRODUCE.md)** — the recipe: weights, patches, serve config, and what to check
  before you trust a number. Start here if you want it *running* rather than explained.
- **[Failure modes](notes/failure-modes.md)** — every failure hit here, organised by what you
  *observe*. Four different causes produce "it loads but the output is wrong".
- **[TODO](notes/TODO.md)** — what is open, ranked, and just as importantly **what is closed and
  why**, so nobody re-opens a dead end on a plausible hunch.
- **[The field](notes/the-field.md)** — who else runs this, what they measured, and which of
  their claims (and ours) did not survive checking.

## Two things that decide whether the model is *usable*, and no speed test can see

Found on 2026-08-29, after a full day of throughput work:

- **Tool calling was rejected outright.** Our launcher set `--reasoning-parser qwen3` and nothing
  else, so every request carrying `tools` returned **HTTP 400**. Fixed with
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml`; now **32/32** across temperatures
  0.2 / 0.6 / 1.0 / default. [Write-up](notes/tool-calling-was-off.md).
- **`--max-model-len 8192` could not hold the model's own reasoning.** A code task emitted 31,115
  characters of thinking before 12,931 of content. 8192 was chosen for benchmarking and was never
  going to serve real work.

Neither is visible to throughput, acceptance, NLL, divergence or coherence tests, because none of
those sends a `tools` field or a long generation. **A serving config has capabilities, not just
speed** — probe both before benchmarking a new recipe.

## What we would not try again

The most useful half of this repo. Each of these looked like a lever and measured null, with the
mechanism understood rather than shrugged at:

- **Hyper-connection quantization or kernels.** 27% of decode GPU time, three interventions
  (blockwise FP8, the CUTE-DSL skinny GEMM, per-channel FP8), all null. They are **latency-bound
  at ~78% of roofline** — a quarter of decode time because there are ~102,000 of them, not because
  any one is expensive. Corroborated independently three ways.
  [Why](notes/why-the-hyper-connections-do-not-respond.md).
- **NVFP4 KV cache** — closed by two independent GB10 measurements plus a structural
  MTP-acceptance penalty, and it fails silently.
- **Lowering `gpu-memory-utilization` to avoid host freezes** — refuted; 0.70 is the worst
  recorded outcome. The cause is absolute free memory at launch, not the ratio.

## Method, earned the hard way

- **The noise floor is 6.9% — for *decode*** (six identical runs: 34.7–37.1 tok/s). **Nothing under
  ~10% is callable from a single run.** ⚠️ **Prefill is far noisier: ±20%.** Three runs of one
  configuration at 8k input spanned 1,633–2,367 tok/s, a 45% range. A prefill claim needs n≥3 and a
  wider bar than a decode claim; treating the 6.9% figure as general is what produced a withdrawn
  batch-size finding on 2026-08-31. This cost us a published claim — "k=2 is the MTP optimum" compared
  a single k=3 run against the *top* of k=2's own spread. Withdrawn.
- **Verify `lfs.sha256`, not file size.** Two size-correct, byte-corrupt shards produced *fluent
  garbage* invariant to every configuration change, and cost a full day plus two retracted
  upstream issues. `aria2` preallocates, so a file reaches full size the moment it starts.
- **Verify the lever is real at the shape level before building anything.**
  [`tools/shapebench.py`](tools/shapebench.py) takes two minutes and would have pre-empted a
  checkpoint build, four failed server starts and three six-run A/B arms.
- **Prove a kernel actually ran.** Log first-sight dispatch keys inside the op — a call-count
  threshold never fires under cudagraph replay. We peeled back *four* layers of "installed but not
  running" before one measurement meant anything.
- **An all-empty cell is not a comparison.** Twice in one session a determinism test reported
  outputs as *identical* when every response was the empty string — the model was still inside
  `<think>` and the token budget ran out, so five empty strings hashed the same and the check
  passed vacuously. A comparison must assert that it compared something: print the character
  counts, and refuse the verdict when the cell is empty. The second occurrence is the instructive
  one — the guard had already been written into the previous script, and **a guard does not travel
  to the next script you write**. It belongs in the harness, not in one copy of it.
- **Clear `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`** when benchmarking a source-level patch;
  a stale compiled graph silently replays your unpatched code. (Config flags *are* hashed
  correctly — we checked before filing.)

## The problem in one table

| build | weights |
|---|---|
| `Qwen/Qwen3.8-Flash-Next` (BF16) | 335.3 GiB |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | **125.9 GiB** |
| **usable on this box** | **~117 GiB** |

It splits cleanly, which is what makes the attempt work: main model 78.2 GiB (196 files), PLE
n-gram table 47.7 GiB (10 files, already FP8). Everything hinges on keeping the n-gram table out
of resident memory — and **the PLE offload is not the bottleneck**: major faults per token *fall
4.4×* from c=1 to c=48, because batched tokens share n-gram rows.

## What the model is

`Qwen4ExpForConditionalGeneration` / `qwen4_exp_text`:

- 48 layers, hidden 2560, **512 experts / 10 active**, `moe_intermediate 640`
- 24 Q heads / 2 KV heads, head_dim 256, **262144** native context
- `layer_types`: 3 × `linear_attention` + 1 × `full_attention`, repeating (GDN hybrid)
- n-gram: 20 M × 2560 = **51.2 B** parameters, 128 shards, attached at layer 1
- **MTP: 1 layer**, "trained with multi-steps" — so k>1 reuses it autoregressively. Qwen publish no
  recommended k; the official vLLM recipe specifies **3**. We previously called **k=2 optimal** on
  decode evidence — **that is withdrawn**: on a fixed-work agent loop k=2 is the *worst*
  configuration measured here (48.6 ms/tok over 3 arms, against 43.6 for no speculation at all),
  while **n=5 is the best** at 31.9. Decode and per-turn latency rank the depths in opposite orders.
  See [which drafter for agent work](notes/which-drafter-for-agent-work.md).

Per-token byte budget (decode, `fp8head`): GDN 1.95 GiB, experts 1.24, hyper-connections 1.19,
QSA 0.59, `lm_head` 0.59, shared_expert 0.44. **The experts are 20% of it at c=1 and ~80% at
c=16** — each sequence pulls its own ten of 512 while the dense path amortizes.

## Layout

    REPRODUCE.md                 the recipe, start to finish  <- start here to build it
    scripts/serve-flashnext.sh   serve config
    tools/shapebench.py          per-shape FP8-vs-BF16 timing, L2 defeated, roofline printed
    patches/                     local vLLM patches + the upstream series
    notes/failure-modes.md       everything that went wrong, by symptom  <- start here
    notes/TODO.md                open work, and what is closed with reasons
    notes/the-field.md           who else runs this, and which claims held up
    notes/log.md                 running record, including the dead ends

    notes/tool-calling-was-off.md            HTTP 400 on every agent request, and the fix
    notes/speculation-on-flash-next.md       what MTP is worth, and what limits it
    notes/why-the-hyper-connections-do-not-respond.md   27% of GPU time, zero to give
    notes/where-the-gpu-time-goes.md         the decode-only kernel profile
    notes/skinny-gemm-on-sm121.md            four blockers, and a null at the end
    notes/the-prefill-decode-confound.md     an explanation of ours that testing refuted
    notes/quantizing-lm-head.md              +11% off, +19.1% under MTP, three blockers
    notes/fp8-mixed-checkpoint.md            the +39% checkpoint switch
    notes/moe-backend-axis.md                the MoE axis, closed -- and what it cost to close
    notes/depth-curve.md                     prefill AND decode are flat to 60k; the QSA signature
    notes/mtp-vs-prefix-cache.md             MTP costs one cache block; the ~68-token break-even
    notes/fp8-kv.md                          x1.72 KV pool, NIAH 5/5, and no speed gain
    notes/temp0-nondeterminism.md            temperature 0 is not reproducible on this model
    notes/results-radixark-vllm.md           the published checkpoint as shipped
    notes/fused-draft-decode-for-qsa.md      a fused draft-decode attempt, and why it was reverted
    notes/upstream-branch.md                 the patch series and where it can go
    notes/single-stream-limit.md             what limits n=1
    notes/load-and-waits.md                  where time goes under concurrency
    notes/ple-access-pattern.md              why the biggest object is a small cost
    notes/fetching-a-slice.md                diffing lfs.sha256 to download 12 GiB, not 123
    notes/choosing-a-quant-scheme.md         picking a scheme when plumbing decides
    notes/quantizing-shared-expert.md        a lever that measures worse than it models
    notes/block-size-is-not-a-kernel-limit.md  a constant, not a kernel, blocks a layer class
    notes/why-no-source-build.md             a file overlay instead of a multi-hour build

## Upstream

| | |
|---|---|
| [#53896](https://github.com/vllm-project/vllm/pull/53896) | `[Model] Support Qwen3.8-Flash-Next` — **MERGED 2026-08-31**. Was for weeks the only place the vLLM implementation existed; it is now on `main`, though not yet in a tagged release. Note the package was renamed `qwen3_8_flash_next` → `qwen4_exp` before merge, so every source path we cite is from the pre-merge build. |
| [#50617](https://github.com/vllm-project/vllm/pull/50617) | fixes the `FP8_PER_CHANNEL_PER_TOKEN` dispatch gap we hit; we added our load-failure evidence rather than opening a duplicate |
| [#53899](https://github.com/vllm-project/vllm/pull/53899) | PLE offload to host memory. Carries `4e8b849b8d97`, the per-request event-pool fix for the [#53960](https://github.com/vllm-project/vllm/issues/53960) startup deadlock — **backported into our venv** ([notes](notes/upstream-branch.md)) |
| [#53960](https://github.com/vllm-project/vllm/issues/53960) | PLE offload deadlocks at warmup (4 reporters). We could not reproduce it and posted the mechanism, the fix pointer and a negative control |
| [#52816](https://github.com/vllm-project/vllm/pull/52816) | DFlash2 — **merged** 2026-08-21 |
| [our branch](https://github.com/jschmied/vllm/tree/gb10-sm121-fixes) | three commits on #53896's head: `quant_config` through `GatedResidual`, `quant_config` on both `ParallelLMHead` sites, and the dispatch fix |

Two contributions from earlier that still stand: a **one-line gate change** so the FP8 PLE is
accepted on an NVFP4 body (correcting checkpoint tables that list RadixArk as not loading — it
does), and **`--cap-add=SYS_PTRACE`** for `VLLM_PLE_CPU_OFFLOAD` in Docker, where
`rebuild_cuda_tensor` needs `pidfd_getfd` and the engine otherwise dies ten minutes in with only
`Failed core proc(s): {}`.

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`): any use, including commercial, with attribution.
Patches under `patches/` that modify vLLM stay under vLLM's Apache 2.0 license; upstream pull
requests carried here are credited in their headers.
