# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Qwen's Qwen4-architecture preview (125B MoE, 6B active, a 51B n-gram table) served from one GB10
with 128 GB of unified memory, on vLLM. This repo is the working record: the recipe, every number
with its data file, the failures by symptom, and the claims of our own we had to withdraw.

**New here?** [notes/what-generalises.md](notes/what-generalises.md) is the short version: five durable
insights, what transferred from the field and what did not, and which of our own conclusions had to be
thrown away. It is synthesis — every number in it points back to the note that carries the data.

**Status (2026-09-26): working, fast, usable** — tool calls, vision, 32K served context (262K-capable).
The stack is vLLM `main` (nightly `1ea7c63f4`) with the PLE table read in place from the checkpoint
([vllm#58439](https://github.com/vllm-project/vllm/pull/58439), ours) instead of the old PLE-offload
worker, MTP n=3 with an NVFP4 drafter, and **RecoverSSM for the GDN layers** (our port, promoted
2026-09-26). One kernel fix of ours is merged into vLLM ([#55180](https://github.com/vllm-project/vllm/pull/55180));
four more are open there.

| what | number | where it comes from |
| --- | --- | --- |
| decode, single stream | **46.5 tok/s** (21.49 ms/tok; 2.53 tokens accepted per verify cycle) — was 17.1 on the published checkpoint | [speed of light §4t](notes/speed-of-light.md), [fp8 checkpoint](notes/fp8-mixed-checkpoint.md), [lm_head](notes/quantizing-lm-head.md), [speculation](notes/speculation-on-flash-next.md) |
| decode, 4 streams | **~100 tok/s** aggregate (99.7–100.0) | [speed of light §4t](notes/speed-of-light.md) |
| agent loop (8 dependent turns, prefix cache + MTP) | **1.10 s/turn**, 1.31 without RecoverSSM | [speed of light §4t](notes/speed-of-light.md) |
| KV capacity in 4 GiB | **103,953 tokens** with RecoverSSM (75,678 without) | [speed of light §4t](notes/speed-of-light.md) |
| distance to the byte floor | 54.2 ms per verify cycle vs a **45.2 ms** floor at 220 GB/s (1.20×) | [speed of light, steps 1–4u](notes/speed-of-light.md) |
| first minutes after a start | ~59 ms/step while the PLE table pages in, 54.7 warm; [vllm#58835](https://github.com/vllm-project/vllm/pull/58835) takes 2.35–2.91 ms/step off the cold window | [speed of light §4e, §4z](notes/speed-of-light.md) |
| TTFT, 7.5k / 29k tokens | 2.6 s / 10.1 s — **previous stack** (dev524 + PLE offload), not re-measured on this one | [prefill findings 117–118](notes/prefill-investigation.md) |
| decode, 16 / 32 streams | ~100 / 110 tok/s aggregate — **previous stack**, not re-measured | [load and waits](notes/load-and-waits.md) |
| greedy determinism | sequential greedy output is reproducible across server restarts: identical hashes in every start of every A/B on this stack. Carried as overlays: deterministic `persistent_topk` ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122), open) and the bit-stable MoE finalize. RecoverSSM changes the text relative to the native GDN path by the size of a summation-order change (first divergence at a median of 26 tokens), and is itself reproducible. Still not batch-invariant under concurrency | [determinism investigation](notes/determinism-investigation.md), [speed of light §4s](notes/speed-of-light.md) |

The single-stream, 4-stream and agent-loop rows are one configuration (prod, measured with KV fixed at
4 GiB so the PLE table stays resident; 2 server starts per arm, ranges not means). The rows marked
*previous stack* are a different configuration and are not comparable. Decode noise start-to-start is
~2 % once the cold window is excluded; prefill is far noisier ([method](notes/method.md)).

## Start here

- **[REPRODUCE.md](REPRODUCE.md)** — weights, the venv overlay, serve config, and what to check before
  you trust a number. Start here to get it *running*.
- **[Speed of light](notes/speed-of-light.md)** — how far decode is from the byte floor, where the rest
  goes, and every lever tried against it (sections 1–4z, newest last).
- **[Failure modes](notes/failure-modes.md)** — every failure hit here, by what you *observe*. Four
  different causes produce "it loads but the output is wrong".
- **[Closed levers](notes/closed-levers.md)** — what looked like a lever and measured null, with the
  mechanism, and the two capability traps no speed test can see (tool calls, context length).
- **[TODO](notes/TODO.md)** — what is open, ranked.
- **[The field](notes/the-field.md)** — who else runs this model, what they measured, and which
  claims (theirs and ours) survived checking.

## Upstream

| | what | state |
| --- | --- | --- |
| [vllm#58439](https://github.com/vllm-project/vllm/pull/58439) | **checkpoint-mapped PLE**: the 47.7 GiB n-gram table is read in place from the safetensors files through a read-only `mmap` (GPUs that read pageable host memory through the host page tables). No table-sized pinned or device allocation; swap use fell from ~50 GiB to 5–6 GiB. Validated on a second Spark by hclsys | ours, open, mergeable, awaiting review |
| [vllm#58835](https://github.com/vllm-project/vllm/pull/58835) | follow-up to #58439: a decode step's cold PLE pages are read with readahead (`MADV_WILLNEED` + `MADV_POPULATE_READ`) instead of a serial touch; cold decode −2.35 / −2.91 ms/step, warm unchanged, outputs identical | ours, opened 2026-09-26 |
| [vllm#56466](https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579) | GDN spec decode + prefix caching (ReplaySSM, WIP): **our RecoverSSM-for-GDN numbers posted as a data point**, asking whether a RecoverSSM-based GDN PR is wanted | comment 2026-09-26 |
| [vllm#55180](https://github.com/vllm-project/vllm/pull/55180) | blockwise-FP8 GEMM on GB10: CTA swizzle restores 150–168 TF at every M (stock collapses to 52) | ours, **merged 2026-09-07** |
| [vllm#55375](https://github.com/vllm-project/vllm/pull/55375) | **MTP output corruption, root-caused here** (findings 126–131): strided PLE conv-state indices. The fix is peakcrosser7's (our duplicate #55467 closed); our evidence and test are on it | **merged 2026-09-05** |
| [vllm#55122](https://github.com/vllm-project/vllm/pull/55122) | deterministic `persistent_topk` (index-ranked ties): greedy prefill reproducible at no end-to-end cost; shipped by blazux as their patch 8 | ours, open |
| [vllm#54912](https://github.com/vllm-project/vllm/pull/54912) | widen the QSA raw-key ring instead of asserting divisibility (`k=5` MTP hard-fails today) | ours, open |
| [vllm#55872](https://github.com/vllm-project/vllm/pull/55872) | LopezCastroRoberto's opt-in deterministic FlashInfer top-k: does not start on sm_121 (tested at their request) | theirs; GB10 result reported |
| [vllm#54521](https://github.com/vllm-project/vllm/issues/54521), [#54928](https://github.com/vllm-project/vllm/issues/54928), [RFC #55394](https://github.com/vllm-project/vllm/issues/55394) | greedy non-determinism evidence; the tile-union QSA prefill RFC | open |
| [vllm#55430](https://github.com/vllm-project/vllm/pull/55430), [#55661](https://github.com/vllm-project/vllm/pull/55661), [#53899](https://github.com/vllm-project/vllm/pull/53899) | tile-union QSA kernel (withdrawn: maintainer wanted > 3 %), the swizzle activation gate, the PLE-offload branch our old stack used | closed |
| [blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3), [MiaAI#4](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/4) | config items and drop-ins for the community recipes | posted |

The posting log with every URL: [notes/upstream/](notes/upstream/README.md). Policy: every post is
drafted in `notes/upstream/`, numbers trace to a finding, AI assistance is disclosed.

## What we found, in one line each

- **The GDN spec-decode kernel's state snapshots were the decode slow spots**: 12 MiB of fp32 state
  written per layer per step sits dirty in the 24 MiB L2, and the next GEMMs pay the write-back
  (`out_proj` 101 → 71 µs clean). RecoverSSM removes them, and in align mode frees 37 % of KV:
  [speed of light §4o, §4s–4u](notes/speed-of-light.md).
- **An fp16 SSM state cache is a quality trade, not free speed** (−2 % / −4 % against a per-token
  perturbation a fifth the size of the 4-bit step): [§4q–4r](notes/speed-of-light.md).
- **Mapping the PLE table beats offloading it**, but a fresh server starts with it paged out: the load
  evicts it, and each decode step then faults ~30 pages the GPU reads one at a time. Readahead fixes
  the cold window; waiting for the host does not help more: [§4e–4j, §4x–4z](notes/speed-of-light.md).
- **"Restart drift" was the cold window**, not noise: measure warm, with KV sized so the table stays
  resident: [§4c, §4k](notes/speed-of-light.md).
- **Three checkpoint levers, one of them ours** (FP8 dense projections +39 %, FP8 `lm_head` +11 %
  and +19 % under MTP), then MTP with an NVFP4 drafter and a 32k draft vocabulary:
  [fp8 checkpoint](notes/fp8-mixed-checkpoint.md), [quantizing lm_head](notes/quantizing-lm-head.md),
  [quantizing the MTP drafter](notes/quantizing-the-mtp-drafter.md).
- **Agent speed is TTFT-bound**, and the decode ranking of drafters inverts on real turns:
  [which drafter](notes/which-drafter-for-agent-work.md), [speculation](notes/speculation-on-flash-next.md).
- **The 24 MiB L2 is the GB10's prefill problem**: the blockwise-FP8 GEMM collapses with M, and a
  scheduler swizzle fixes it bit-identically (merged as #55180): [findings 95/100/102](notes/prefill-investigation.md).
- **Temperature 0 is not reproducible under concurrency**, and the causes are kernels, not the
  drafter: [determinism investigation](notes/determinism-investigation.md), [temp0](notes/temp0-nondeterminism.md).
- **Nulls with mechanism** — hyper-connections (latency-bound), NVFP4 KV, skinny GEMM, a small-M BF16
  kernel, streaming stores, the verify kernel's launch config: [closed levers](notes/closed-levers.md),
  [speed of light §3c, §4l, §4p, §4v](notes/speed-of-light.md).

## Layout

    REPRODUCE.md                  the recipe, start to finish
    scripts/serve-flashnext.sh    serve config (identical to the one prod runs)
    tools/main/main1ea7-prod-overlay.diff   the whole prod venv overlay against the pristine nightly
    tools/rssm/                   RecoverSSM for GDN: kernels, backends, patch scripts, tests (defer/: next step)
    tools/plecold/                PLE cold-window instruments and the readahead fill
    tools/prof/                   nsys / torch-profile analysis (nsyscmp.py compares two traces)
    tools/armrun.py               the A/B runner every server number comes from
    notes/speed-of-light.md       decode vs the byte floor, sections 1–4z
    notes/prefill-investigation.md   numbered findings (prefill, kernels, cache, the mapped PLE)
    notes/determinism-investigation.md   greedy reproducibility; starts with an "answers by question" index
    notes/upstream/               drafts of every post and the posting log
    notes/data/                   raw logs behind every number

Per-topic notes ([model and memory budget](notes/model-and-memory-budget.md),
[fp8 KV](notes/fp8-kv.md), [single-stream limit](notes/single-stream-limit.md),
[fetching a slice](notes/fetching-a-slice.md), …) are listed in [notes/](notes/) and linked from the findings.

Hardware: NVIDIA DGX Spark, GB10, sm_121, 128 GB unified, aarch64.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`): any use, including commercial, with attribution.
Patches under `patches/` and `tools/` that modify vLLM stay under vLLM's Apache 2.0 license; upstream
pull requests carried here are credited in their headers.
