# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Qwen's Qwen4-architecture preview (125B MoE, 6B active, a 51B n-gram table) served from one GB10
with 128 GB of unified memory, on vLLM. This repo is the working record: the recipe, every number
with its data file, the failures by symptom, and the claims of our own we had to withdraw.

**Status: working, fast, and usable** — 262K-capable context, tool calls, vision, and two kernel
fixes of ours now in vLLM or under review there.

| what | number | where it comes from |
| --- | --- | --- |
| decode, single stream | 17.1 → **36.5 tok/s** (checkpoint levers + MTP) | [speculation](notes/speculation-on-flash-next.md), [fp8 checkpoint](notes/fp8-mixed-checkpoint.md), [lm_head](notes/quantizing-lm-head.md) |
| decode, 16 / 32 streams | ~100 / 110 tok/s aggregate | [load and waits](notes/load-and-waits.md) |
| TTFT, 7.5k / 29k tokens | **2.6 s / 10.1 s** (≈ 2,800 tok/s prefill) | [prefill findings 117–118](notes/prefill-investigation.md) |
| warm agent turn (prefix cache + MTP) | **1.5 s** per 130-token turn, from 2.05 | [finding 94](notes/prefill-investigation.md), [mtp vs prefix cache](notes/mtp-vs-prefix-cache.md) |
| greedy determinism | reproducible sequentially after three kernel fixes; not under concurrency | [determinism](notes/determinism-investigation.md) |

Decode numbers are a different configuration from the prefill ones and not comparable across rows;
each link says how its number was produced. Nothing here is quoted from one run: decode noise is
6.9 %, prefill ±20 %, so claims carry two or three server starts ([method](notes/method.md)).

## Start here

- **[REPRODUCE.md](REPRODUCE.md)** — weights, patches, serve config, and what to check before you
  trust a number. Start here to get it *running*.
- **[Failure modes](notes/failure-modes.md)** — every failure hit here, by what you *observe*. Four
  different causes produce "it loads but the output is wrong".
- **[Closed levers](notes/closed-levers.md)** — what looked like a lever and measured null, with the
  mechanism, and the two capability traps no speed test can see (tool calls, context length).
- **[TODO](notes/TODO.md)** and **[prefill plan](notes/prefill-plan.md)** — what is open, ranked, and
  what is closed and why.
- **[The field](notes/the-field.md)** — who else runs this model, what they measured, and which
  claims (theirs and ours) survived checking.

## Upstream

| | what | state |
| --- | --- | --- |
| [vllm#55430](https://github.com/vllm-project/vllm/pull/55430) | tile-union QSA prefill kernel: consecutive rows share one K/V gather; −2.8 % TTFT at 8k, −1.7 % at 30k on SM121 | PR, review |
| [vllm#55394](https://github.com/vllm-project/vllm/issues/55394) | the RFC behind it: design, GB10 numbers, bring-up table for other parts | open |
| [vllm#55122](https://github.com/vllm-project/vllm/pull/55122) | deterministic `persistent_topk` (index-ranked ties): greedy prefill reproducible at no cost; shipped by blazux as their patch 8 | PR, review |
| [vllm#55180](https://github.com/vllm-project/vllm/pull/55180) | blockwise-FP8 GEMM on GB10: CTA swizzle restores 150–168 TF at every M (stock collapses to 52) | PR, review |
| [vllm#54521](https://github.com/vllm-project/vllm/issues/54521), [#54928](https://github.com/vllm-project/vllm/issues/54928) | greedy non-determinism: the three GB10 causes, the batch-shape channel, the GEMM M-invariance table | evidence posted |
| [vllm#53899](https://github.com/vllm-project/vllm/pull/53899) | PLE offload worker: GB10 validation, the `--cap-add=SYS_PTRACE` and KV-profiling notes | validated |
| [blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3), [MiaAI#4](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/4) | the config items and drop-ins for the community recipes | posted |

Earlier items and the posting log: [notes/upstream/](notes/upstream/README.md),
[history](notes/upstream/history.md). Policy: every post is drafted in `notes/upstream/`, numbers
trace to a finding, AI assistance is disclosed.

## What we found, in one line each

- **Three checkpoint levers, one of them ours** (FP8 dense projections +39 %, FP8 `lm_head` +11 %
  and +19 % under MTP, then MTP): [fp8 checkpoint](notes/fp8-mixed-checkpoint.md),
  [quantizing lm_head](notes/quantizing-lm-head.md).
- **Agent speed is TTFT-bound**, and the decode ranking of drafters inverts on real turns:
  [which drafter](notes/which-drafter-for-agent-work.md), [speculation](notes/speculation-on-flash-next.md).
- **The PLE offload is not the bottleneck**: faults per token fall 4.4× under load;
  [ple access pattern](notes/ple-access-pattern.md), [load and waits](notes/load-and-waits.md).
- **Warm turns paid a full re-prefill past the 1,600-token align block**; the trailing-block flag
  and an aligned prefix fix most of it: [finding 94/97/98](notes/prefill-investigation.md),
  [prefix cache is not reuse](notes/prefix-cache-is-not-reuse.md).
- **The 24 MiB L2 is the GB10's prefill problem**: the blockwise-FP8 GEMM collapses with M, a
  scheduler swizzle fixes it bit-identically: [findings 95/100/102](notes/prefill-investigation.md).
- **QSA prefill rows share 87–94 % of their selected blocks**; the tile-union kernel gathers once
  per tile: [findings 92–119](notes/prefill-investigation.md).
- **Temperature 0 is not reproducible under concurrency**, and the causes are kernels, not the
  drafter: [determinism investigation](notes/determinism-investigation.md),
  [temp0](notes/temp0-nondeterminism.md).
- **Nulls with mechanism** — hyper-connections (latency-bound at 78 % of roofline), NVFP4 KV,
  skinny GEMM, MoE backend, shared-expert quant: [closed levers](notes/closed-levers.md).

## Layout

    REPRODUCE.md                  the recipe, start to finish
    scripts/serve-flashnext.sh    serve config
    tools/                        probes and microbenchmarks (shapebench, gemm_m_invariance, qsa_union_*)
    tools/main/                   env-gated patches for the nightly venv, the overlay diffs, memguard
    patches/                      local vLLM patches and the upstream series
    notes/prefill-investigation.md   numbered findings 1–119 (prefill, kernels, cache)
    notes/determinism-investigation.md   findings on greedy reproducibility
    notes/upstream/               drafts of every post, the posting log, the PR plan
    notes/data/                   raw logs behind every number
    notes/log.md                  running record, including the dead ends

Per-topic notes ([model and memory budget](notes/model-and-memory-budget.md),
[depth curve](notes/depth-curve.md), [fp8 KV](notes/fp8-kv.md), [single-stream limit](notes/single-stream-limit.md),
[why no source build](notes/why-no-source-build.md), [fetching a slice](notes/fetching-a-slice.md), …)
are listed in [notes/](notes/) and linked from the findings.

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`): any use, including commercial, with attribution.
Patches under `patches/` that modify vLLM stay under vLLM's Apache 2.0 license; upstream pull
requests carried here are credited in their headers.
