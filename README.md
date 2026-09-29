# Qwen3.8-Flash-Next on a DGX Spark (GB10)

Qwen's Qwen4-architecture preview (125B MoE, 6B active, a 51B n-gram table) served from one GB10
with 128 GB of unified memory, on vLLM. This repo is the working record: the recipe, every number
with its data file, the failures by symptom, and the claims of our own we had to withdraw.

**New here?** [notes/what-generalises.md](notes/what-generalises.md) is the short version: five durable
insights, what transferred from the field and what did not, and which of our own conclusions had to be
thrown away. It is synthesis — every number in it points back to the note that carries the data.

**Status (2026-09-28): working, fast, usable** — tool calls, vision, 32K served context (262K-capable).
The stack is vLLM `main` (nightly `1ea7c63f4`) with the PLE table read in place from the checkpoint
([vllm#58439](https://github.com/vllm-project/vllm/pull/58439), ours) instead of the old PLE-offload
worker, **RecoverSSM for the GDN layers** as submitted upstream ([vllm#58863](https://github.com/vllm-project/vllm/pull/58863),
ours, enabled by `--use-replayssm`), and **MTP K=5 with probabilistic drafting** over a 32k-token NVFP4 draft
head. One kernel fix of ours is merged into vLLM ([#55180](https://github.com/vllm-project/vllm/pull/55180));
five more are open there.

> **Config change 2026-09-27:** K 3 → 5 (needs `--block-size 1728`), probabilistic drafting, and the PR form of
> RecoverSSM replace the 09-26 config. It is installed and validated on a server with the exact prod launcher and drop-ins
> (every path line present; code 15.37 ms/tok, prose 24.72, 4 streams 134.9 tok/s; [data](notes/data/prodval/)).
> **2026-09-28, loading:** the fast-loading set is in prod's venv: [vllm#58868](https://github.com/vllm-project/vllm/pull/58868)
> plus blazux's expert name index and MTP name prefilter (ported). A cold start's model loading drops from ~11 min to under
> 1 min (605–651 → 56–57 s over 2 starts per arm, identical greedy hash; [§5w](notes/speed-of-light.md)). Loading only; a start with prod's exact config and venv is ready in **2 min 39 s** (was ~12 min).
> The whole 09-28 config (bf16 SSM state, `--prefix-match-unit 64`, fast loading, tool guards) is validated on one start:
> every path line present, code 14.74 ms/tok, prose 23.66, 4 streams 144.8 tok/s, cache replay equal
> ([§5y](notes/speed-of-light.md), [data](notes/data/prodval2/)).
> **2026-09-28, HC fusion + F4 in prod** (user: "yes, both goes to prod"; drop-in 55, installed, service still stopped):
> TTFT −5.5…−7.9 % at 8k / −6.2…−7.8 % at 30k, decode c=1 −1…−2 % on the clone venv ([§5aa](notes/speed-of-light.md),
> [§5z](notes/speed-of-light.md)). **Validated on the prod venv** (prodval3, [§5ac](notes/speed-of-light.md)): every path line present,
> TTFT 2.73 / 9.75 s at 8k / 30k (−5.4 / −6.2 % vs the morning config), code 14.60 ms/tok, outputs unchanged; GSM8K/HumanEval
> screen unchanged (McNemar p ≥ 0.25). The default KV pool shrinks 5.9 % (full-graph memory). All headline rows re-measured on this config ([§5ad](notes/speed-of-light.md)).
> K=5 is the better default for code-heavy agent work; prose is 5 % slower than at K=3 ([§5l](notes/speed-of-light.md)).
> Forwarded to nightly `a9eafde59` (266 commits newer; [§5s](notes/speed-of-light.md)): the overlay applies and c=1 output is
> bit-identical to prod, but c=4 is 4.5 % slower and not reproducible, so prod stays on `1ea7c63f4` for now.

| what | number | where it comes from |
| --- | --- | --- |
| decode, single stream, **code** (4 prompts, 700 tokens, first pass) | **68.1–68.4 tok/s** greedy (14.63–14.68 ms/tok, 4.37 accepted per verify cycle), **58.8–61.4** sampled — 65.7 / 62.4 on 09-27, 59.0 / 56.0 on 09-26 | [speed of light §5ad](notes/speed-of-light.md), [§5l, §5n, §5o](notes/speed-of-light.md) |
| decode, single stream, **prose** (same probe) | **42.2–42.4 tok/s** greedy — 41.1 on 09-27, 43.1 on 09-26 (K=3 is better on prose) | [speed of light §5ad, §5o](notes/speed-of-light.md) |
| decode, single stream, 09-26 config (warm second pass, prose) | 46.5 tok/s (21.49 ms/tok; 2.53 accepted per cycle) — was 17.1 on the published checkpoint | [speed of light §4t](notes/speed-of-light.md), [fp8 checkpoint](notes/fp8-mixed-checkpoint.md), [lm_head](notes/quantizing-lm-head.md), [speculation](notes/speculation-on-flash-next.md) |
| precision options: bf16 SSM state (**adopted** 2026-09-28); GDN projections as NVFP4 W4A16 (not adopted) | NVFP4 GDN: code 71.3 tok/s greedy (−7 % decode) but **+6 % TTFT**. Both pass a GSM8K + HumanEval screen and **SWE-bench** (58 instances × 2 runs: prod 48/52, NVFP4 GDN 51/50, bf16 state 50/51 — all inside prod's own spread); the bf16 state is in the prod config since 2026-09-28 | [speed of light §5j, §5o, §5r, §5u](notes/speed-of-light.md) |
| **in prod since 2026-09-28:** fused hyper-connection kernels (`FN_HCFUSE=1`) | **TTFT −5.5…−7.9 % at 8k, −6.2…−7.8 % at 30k**; warm agent turns ~−1 %; decode unchanged; deterministic 1-ulp drift, GSM8K/HumanEval screen unchanged (McNemar p ≥ 0.25; 2 starts per arm). The byte ledger found the HC normalized copy and gate round-tripping DRAM at prefill | [speed of light §5aa](notes/speed-of-light.md) |
| **in prod since 2026-09-28:** full CUDA graphs for the RecoverSSM verify path (F4) | code c=1 **−1.1…−2.1 %**, cache-hit replay and TTFT −1…−2.5 %, prose and 4 streams null; outputs identical at c=1 (2 starts per arm; a first "8k regression" was a probe artefact, withdrawn) | [speed of light §5z](notes/speed-of-light.md) |
| **prod candidates, measured, not installed (2026-09-29):** Triton NVFP4 prefill MoE (`FN_MOEFUSE=1`) + fused GDN norm/FP8 quant (`FN_GDNNQ=1`) | together: **TTFT 2.56 s at 8k / 9.12 s at 30k** (−6.0 % / −6.1…−6.6 % vs prod's 2.73 / 9.73–9.76 on the same day; ~2,930 / ~3,210 tok/s prefill), decode per cycle null. The MoE kernels are bit-identical to FlashInfer's (outputs unchanged, −3.8…−4.3 % at 30k alone); GDNNQ drifts the text, GSM8K/HumanEval unchanged. Cache-hit replay with GDNNQ open (its decoded text differs) | [speed of light §5ag, §5af](notes/speed-of-light.md) |
| decode, 4 streams | **code, K=5: 144.3–144.7 tok/s** aggregate; prose 80.8–83.6. The 09-26 config on the decode probe: ~100 | [speed of light §5ad, §4t](notes/speed-of-light.md) |
| agent loop (8 dependent turns, prefix cache + MTP) | **1.25 s/turn** (K=5, short prose-like turns); 1.10 on the 09-26 K=3 config, 1.31 without RecoverSSM | [speed of light §5ad, §4t](notes/speed-of-light.md) |
| warm agent turns (46 replayed SWE-bench turns) | **0.563–0.567 s median** with `--prefix-match-unit 64` (0.87 without it; recompute −74 %) | [speed of light §5ad, §5t](notes/speed-of-light.md) |
| KV capacity in 4 GiB | **105,325 tokens** in the prod config (K=5, block 1728, bf16 SSM state; re-confirmed 09-28); 77,608 with the fp32 state. 09-26 config (K=3): 103,953 with RecoverSSM, 75,678 without. At prod's default size: 791,113 (840,265 before full CUDA graphs) | server logs of §5ad, §5ac; [speed of light §4t](notes/speed-of-light.md) |
| distance to the byte floor | K=5, fp32 SSM state (09-27): 66.1 ms per verify cycle on code vs a **~53–54 ms** floor at 220 GB/s (1.21–1.25×, estimate: the 6-row expert union is interpolated, not captured; floor ≈ 80 tok/s on code). K=3 (09-26): 54.2 vs 45.2 ms (1.20×, measured routing) | [speed of light §2a, §5q](notes/speed-of-light.md) |
| first minutes after a start | ~59 ms/step while the PLE table pages in, 54.7 warm (09-26 config); [vllm#58835](https://github.com/vllm-project/vllm/pull/58835) takes 2.35–2.91 ms/step off the cold window | [speed of light §4e, §4z](notes/speed-of-light.md) |
| TTFT, ~7.5k / ~29k tokens, cold (unique prompt) | **2.71–2.72 s / 9.69–9.73 s** on the 09-28 prod config (HC fusion); 2.77 / 10.25–10.33 with the bf16 state before it; 2.6 / 10.1 on the previous stack | [speed of light §5ad, §5aa, §5r](notes/speed-of-light.md), [prefill findings 117–118](notes/prefill-investigation.md) |
| cold start, model loading (page cache dropped) | **56–57 s** with the fast-loading set (main weights 41.6–41.9 s, MTP drafter 3.5–3.75 s; 2 starts); 111–121 s with #58868 alone; **575–651 s stock** | [speed of light §5w](notes/speed-of-light.md) |
| decode, 8 / 16 streams (prose) | **122.2–123.1 / 161.9–165.0 tok/s** aggregate; 32 streams not served (`--max-num-seqs 16`). The previous stack did ~100 / 110 at 16 / 32 | [speed of light §5ad](notes/speed-of-light.md), [load and waits](notes/load-and-waits.md) |
| greedy determinism | sequential greedy output is reproducible across server restarts: identical hashes in every start of every A/B on this stack. Carried as overlays: deterministic `persistent_topk` ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122), open) and the bit-stable MoE finalize. RecoverSSM changes the text relative to the native GDN path by the size of a summation-order change (first divergence at a median of 26 tokens), and is itself reproducible. Still not batch-invariant under concurrency | [determinism investigation](notes/determinism-investigation.md), [speed of light §4s](notes/speed-of-light.md) |

The code and prose rows use the code probe of §5l: each prompt is seen once per server, so they include the cold
PLE pages (≈7 % above a warm second pass); compare them with each other, not with the 46.5 row. All rows above were re-measured on the 09-28 prod config (§5ad) unless they name another config. Unless a row says otherwise,
KV is fixed at 4 GiB so the PLE table stays resident, with 2 server starts per arm, ranges not means. The rows marked
*previous stack* are a different configuration and are not comparable. Decode noise start-to-start is
~2 % once the cold window is excluded; prefill is far noisier ([method](notes/method.md)).

## Start here

- **[REPRODUCE.md](REPRODUCE.md)** — weights, the venv overlay, serve config, and what to check before
  you trust a number. Start here to get it *running*.
- **[Speed of light](notes/speed-of-light.md)** — how far decode is from the byte floor, where the rest
  goes, and every lever tried against it (sections 1–5ad, newest last).
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
| [vllm#58863](https://github.com/vllm-project/vllm/pull/58863) | **RecoverSSM for Qwen GDN and the PLE short conv**: verify from a read-only checkpoint with a per-token replay record, commit once after sampling; no per-draft Mamba blocks. Shares Kimi-K3's commit kernels; also fixes the FULL_AND_PIECEWISE fallback dropping to no CUDA graphs under breakable graphs. c=4 −5.2 % per cycle, agent turn −16 %, +37 % KV | ours, opened 2026-09-27 |
| [vllm#58835](https://github.com/vllm-project/vllm/pull/58835) | follow-up to #58439: a decode step's cold PLE pages are read with readahead (`MADV_WILLNEED` + `MADV_POPULATE_READ`) instead of a serial touch; cold decode −2.35 / −2.91 ms/step, warm unchanged, outputs identical | ours, opened 2026-09-26 |
| [vllm#58868](https://github.com/vllm-project/vllm/pull/58868#issuecomment-5865989952) | Willian-Zhang's mmap page-fault fix for weight loading on integrated GPUs: **our GB10 TP1 A/B posted** (main load 505–534 → 65–66 s, identical output); in our prod venv | theirs, open; data point 2026-09-28 |
| [vllm#56466](https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579) | GDN spec decode + prefix caching (ReplaySSM, WIP): **our RecoverSSM-for-GDN numbers posted as a data point**, asking whether a RecoverSSM-based GDN PR is wanted | comment 2026-09-26 |
| [vllm#55180](https://github.com/vllm-project/vllm/pull/55180) | blockwise-FP8 GEMM on GB10: CTA swizzle restores 150–168 TF at every M (stock collapses to 52) | ours, **merged 2026-09-07** |
| [vllm#55375](https://github.com/vllm-project/vllm/pull/55375) | **MTP output corruption, root-caused here** (findings 126–131): strided PLE conv-state indices. The fix is peakcrosser7's (our duplicate #55467 closed); our evidence and test are on it | **merged 2026-09-05** |
| [vllm#55122](https://github.com/vllm-project/vllm/pull/55122) | deterministic `persistent_topk` (index-ranked ties): greedy prefill reproducible at no end-to-end cost; shipped by blazux as their patch 8 | ours, open |
| [vllm#54912](https://github.com/vllm-project/vllm/pull/54912) | widen the QSA raw-key ring instead of asserting divisibility (`k=5` MTP hard-fails at the auto block size; `--block-size 1728` is the workaround prod uses) | ours, open |
| [vllm#55872](https://github.com/vllm-project/vllm/pull/55872) | LopezCastroRoberto's opt-in deterministic FlashInfer top-k: does not start on sm_121 (tested at their request) | theirs; GB10 result reported |
| [vllm#54521](https://github.com/vllm-project/vllm/issues/54521), [#54928](https://github.com/vllm-project/vllm/issues/54928), [RFC #55394](https://github.com/vllm-project/vllm/issues/55394) | greedy non-determinism evidence; the tile-union QSA prefill RFC | open |
| [vllm#55430](https://github.com/vllm-project/vllm/pull/55430), [#55661](https://github.com/vllm-project/vllm/pull/55661), [#53899](https://github.com/vllm-project/vllm/pull/53899) | tile-union QSA kernel (withdrawn: maintainer wanted > 3 %), the swizzle activation gate, the PLE-offload branch our old stack used | closed |
| [blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3), [MiaAI#4](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/4) | config items and drop-ins for the community recipes | posted |

The posting log with every URL: [notes/upstream/](notes/upstream/README.md). Policy: every post is
drafted in `notes/upstream/`, numbers trace to a finding, AI assistance is disclosed.

## What we found, in one line each

- **The text decides the draft depth**: K=5 is −10 % on code (4.35 tokens per cycle) and +5 % on prose;
  probabilistic drafting over the 32k NVFP4 draft slice adds −5.5 % on sampled code and is exact (the slice's
  logits are scattered into a −inf full-vocabulary buffer, so tokens outside it have q = 0):
  [§5l, §5n, §5o](notes/speed-of-light.md).
- **The GDN spec-decode kernel's state snapshots were the decode slow spots**: 12 MiB of fp32 state
  written per layer per step sits dirty in the 24 MiB L2, and the next GEMMs pay the write-back
  (`out_proj` 101 → 71 µs clean). RecoverSSM removes them, and in align mode frees 37 % of KV:
  [speed of light §4o, §4s–4u](notes/speed-of-light.md).
- **An fp16 SSM state cache is a quality trade, not free speed** (−2 % / −4 % against a per-token
  perturbation a fifth the size of the 4-bit step): [§4q–4r](notes/speed-of-light.md).
- **Mapping the PLE table beats offloading it**, but a fresh server starts with it paged out: the load
  evicts it, and each decode step then faults ~30 pages the GPU reads one at a time. Readahead fixes
  the cold window; waiting for the host does not help more: [§4e–4j, §4x–4z](notes/speed-of-light.md).
- **Weight loading was page faults inside the driver copy**: touching the mmap'd pages on the CPU first (vllm#58868),
  an expert name index and an MTP name prefilter take model loading from ~11 min to under 1 (identical output):
  [§5w](notes/speed-of-light.md).
- **The Mamba prefix cache is sparse by design**: states are kept only at prompt ends and shared-prefix junctions, so
  a prompt's first repeat misses and at the default a decode-written state is not kept (measured: 0 of 16 follow-ups reused one); with the interval set,
  a served decode-written state did not contaminate the output (vllm#53912):
  [§5x](notes/speed-of-light.md).
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
    tools/main/dropins/           the systemd drop-ins prod runs (env for the launcher)
    tools/rssm/                   RecoverSSM for GDN: kernels, backends, patch scripts, tests (defer/: next step)
    tools/ksweep/, tools/dprob/, tools/stack/   MTP depth on code vs prose, probabilistic drafting, the stacked A/B
    tools/plecold/                PLE cold-window instruments and the readahead fill
    tools/prof/                   nsys / torch-profile analysis (nsyscmp.py compares two traces)
    tools/armrun.py               the A/B runner every server number comes from
    notes/speed-of-light.md       decode vs the byte floor, sections 1–5ad
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
