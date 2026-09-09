> **Superseded, 2026-08-27.** This page opened by saying no other project was a single-box vLLM
> deployment, "which is the only reason this repo exists". That stopped being true within a day.
> `starkweatherdigital` and `getrefined` both run vLLM on a Spark, `0xBakeer` added a vLLM recipe
> alongside their llama.cpp one, and `SirTificate` runs vLLM on 2x RTX PRO 6000. The field went
> from ~9 projects to ~35 in 48 hours. Treat every count and ranking below as a snapshot.
>
> What still distinguishes this repo is narrower and worth stating honestly: a kernel-level
> profile of where single-stream time actually goes on GB10, concurrency measured past c=8, and
> the failure-mode catalogue. Not the deployment itself.


## Who is in the field, and what each is worth reading for

| project | stack | hardware | read it for |
|---|---|---|---|
| [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | SGLang TP2, NVFP4 | **2×** Spark | 64 tok/s single-stream; the `!!!!!!` tool-call loop; the `finish_reason: "length"` diagnostic |
| [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp **and** vLLM | 1× Spark | prefix caching 1.76×; the 1,600-block boundary; build-sha stamping; three self-retractions |
| [DJLougen](https://github.com/DJLougen/Qwen3.8-Flash-Next-One-DGX-Spark) | llama.cpp + QSA patch | 1× Spark | QSA kernels doubling long-context decode; output-hash locking; PLE residency **no-win** |
| [spark-arena `e9307821`](https://spark-arena.com) | vLLM, NVFP4 PLE | 1× Spark | the most comparable external run; an NVFP4 PLE checkpoint that loads |
| [veloGB10](https://github.com/sf-stav/veloGB10) | Rust + hand-written PTX | 1× Spark | the *lossless-MTP contract*; asymmetric int8-K/q4-V cache; **no Flash-Next support** |
| [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) | vLLM container | 1× Spark | the image most of the field builds from |

Cross-cutting method the field runs and we did not: **build-sha stamping** (0xBakeer),
**output-hash locking** (DJLougen), **kernel goldens** (veloGB10), and **release-as-measurement-epoch**
versioning. See the 2026-08-30 entries.

## What the dual-Spark deployment shows

Same checkpoint this repo uses (`RadixArk/Qwen3.8-Flash-Next-NVFP4`). Per node:

    GPU weights            ~62.5 GB
    pinned host PLE table  ~11   GB
    KV cache (956,800 tok)   9.0 GB
    free                   ~17.5 GB

Two things follow. **TP2 halves the weights**, which is how they sidestep the residency
problem rather than solve it — not a technique available on one box. And their pinned PLE
is ~11 GB per node, ~22 GB total, well under the 47.7 GiB table; whether that is sharding,
partial pinning, or a staging buffer is worth reading their `start.sh` for before designing
anything here.

Their KV figure is a useful reassurance either way: 956,800 tokens in 9.0 GB, because the
model has 2 KV heads and mostly linear attention.

## The SM121 risk, and why vLLM appears to dodge it

MiaAI had to patch SGLang so Qwen Sparse Attention falls back to Triton FlashDecoding when
`is_sm100_supported()` is false. vLLM PR #53896 ships that by construction:

```
nvidia/qsa.py:3    """NVIDIA QSA owner with Triton kernels."""
nvidia/qsa.py:74   return "QWEN4_EXP_QSA_TRITON"

nvidia/ops/qsa.py:791-792
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
```

QSA is Triton from the start, and the faster path is explicitly excluded for the 120 family —
which is sm_121. This was the largest unknown for a first boot here and it looks addressed
deliberately, not by accident. `low_latency_gemm.py:83` similarly gates a decode GEMM on
device capability (10, 3), so that path is simply not taken on GB10.

**Unverified.** Nothing here has been booted yet; this is code reading, not a result.


## 2026-08-26 21:18 — the agent `!!!!!!` loop (and why it matters here)

MiaAI documented [sgl-project/sglang#36537](https://github.com/sgl-project/sglang/issues/36537):
with **thinking on + OpenAI `tools` + `--tool-call-parser qwen3_coder`**, the server emits
**token ID 0** in a tight loop. This tokenizer decodes 0 as `!`, so the reply becomes
`!!!!!!…` until `max_tokens`; speculative accept rate falls to 0.00 and disconnected clients
keep generating. Their workaround is to turn thinking off for those sessions
(`chat_template_kwargs: {"enable_thinking": false}`), and to cap agent temperature at ≤0.7 —
a residual loop was seen at 1.0. Without the parser, tool calls leak as `<tool_call>` XML in
`content` instead of `message.tool_calls`. Their conclusion:

> There is no day-0 flag that gives thinking *and* structured tools together.

⚠️ **Settled since (2026-08-29): 32/32 tool calls across temperatures with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, thinking on.** The framing below — that this was the most important open question for this repo — was true when written and is not now. The justification for a vLLM route
is agent traffic with concurrency. If thinking and tools cannot coexist, the agent case is
damaged whatever the stack. What is not yet known is whether the fault is SGLang's tool-call
parser or the model's chat template — and that is testable here, on a different stack with a
different parser (`qwen3_xml`). **That test is worth more than a throughput number**, and it is
now the first thing to try after the model loads.

Prior symptoms of the same shape on this box, for whoever picks this up: temperature 1.0
breaking tool-calling on Qwen3.6-35B-A3B, and `<tool_call>` leaking unparsed when a
chat-template-file model is served without `jinja=true`. The pattern is not new to this model.


## 2026-08-26 21:20 — 0xBakeer corrects their own numbers, and finds the real problem

Three corrections worth carrying, from their `bench/run_bench.py` and `cold_vs_warm.sh`:

1. **97.4 tok/s was one unrepresentative run.** Corrected to **52.6 cold / 74.6 warm** on
   copy-heavy work. Free-form prose is 22.1 tok/s at 5.8% draft acceptance.
2. **Speculation does help after all** — the earlier "no speedup" was measured with warming
   off. With `ngram-mod`, warming the table is worth up to **+42%** on copy-heavy work,
   because verifying a 50-60 token span touches many n-gram rows at once. Their earlier claim
   is annotated rather than deleted, which is the right way to do it.
3. **The page cache does not hold under load.**

   | state | table cached |
   |---|---|
   | after the startup warm | 100% |
   | after a benchmark pass | **0.1%** |
   | after re-warming, under load | 50% |
   | dropped caches, re-warmed, box idle | 99% |

   At 262k context the model's own file pages evict the embedding table. A full warm is one
   sequential 26.8 GiB read (~26 s at ~1.0 GiB/s).

**This is the finding that matters most for the vLLM route here.** The plan was to keep the
PLE non-resident behind the page cache. Point 3 says that is not a steady state on one box —
weights and table compete for the same unified pool and the weights win. mmap buys a warm
start that decays under exactly the load we care about.

It does not kill the idea, but it changes what a solution has to look like: something that
*pins* a working subset rather than trusting the kernel's eviction policy. Which is also the
open question about MiaAI's ~11 GB pinned PLE per node — that number now looks less like a
staging buffer and more like a deliberate hot-set. Reading their `start.sh` moved up the list.


## 2026-08-27 — single-box vLLM is not impossible, and there is a specific bug in the way

**Correction to this repo's earlier reasoning.** MiaAI's `start.sh` header says "135 GB of
weights does not fit one 128 GB Spark", and that was read here as arithmetic ruling out a
single box. It does not — their figure is for a **PLE-resident** deployment. With
`VLLM_PLE_CPU_OFFLOAD=1` the resident model is far smaller.

[vllm#53960](https://github.com/vllm-project/vllm/issues/53960) (`jdmays13`) runs precisely the
configuration this repo targets — `RadixArk/Qwen3.8-Flash-Next-NVFP4`, GB10 sm_121, **TP=1**,
PLE offloaded — and gets all the way through init:

    [model_runner.py:407]  Model loading took 80.28 GiB memory and 654.0 s
    [connector.py:231]     PleOffload: registered 1 PleOffloadLayer(s) (ipc:///tmp/...)
    [kv_cache_utils.py]    GPU KV cache size: 271,610 tokens

80.28 GiB resident, a 271k-token KV cache, FlashInfer autotune complete. Then it **hangs
permanently at CUDA-graph warmup**, 3/3 reproductions (25, 60, 60 minutes).

### It is not a GB10 problem

`jhsmith409` reproduced it byte-for-byte on a single RTX PRO 6000 Blackwell Max-Q — **sm_120,
x86_64, 96 GB discrete memory**, `max_model_len=65536`, no speculative decoding. Their summary:
"It is not sm_121, aarch64, unified memory, or MTP."

### The mechanism the two stacks suggest

    MainThread (100% CPU):     replay (torch/cuda/graphs.py:186)
                               __call__ (vllm/compilation/cuda_graph.py:360)
                               warmup_kernels (vllm/v1/worker/gpu/warmup.py:330)

    "ple-offload-dp0" (idle):  get (queue.py:171)   block=True, timeout=None
                               _request_loop (vllm/v1/ple_offload/connector.py:263)

The offload worker is blocked on an **empty** queue while the main thread spins inside CUDA
graph replay. That is consistent with a host-side blocking IPC round-trip having been captured
into a CUDA graph: the graph cannot service it, so the request never reaches the worker and
both sides wait forever.

**Testable prediction:** `--enforce-eager` (no graph capture) should not hang. If it serves,
the bug is graph capture of the PLE path, not the offload mechanism — and the workaround is
available to everyone blocked on this today. That is now the first experiment here, ahead of
throughput and ahead of any streaming-loader work.

**Also of note:** there is an image `vllm/vllm-openai:qwen38-flash-next`
(`sha256:fc120ece0a38`, vllm `0.1.dev20073+g8e685d198`) — this path is further along than
PR #53896 alone suggests.

## 2026-08-29 05:09 — 0xBakeer corrects five more claims; none of ours depend on them

Their commit "Fix five claims a review pass caught, one of them an arithmetic error" retracts:
a stale `1.26x` MTP-on-prose figure repeated in four files next to the `27.8 -> 32.2 = 1.16x`
that disproves it; a "12% prefill falloff" that conflated endpoints (2%) with the peak at 20k
(11%); a "~4 min to read 128k" that assumed a flat ~540 tok/s prefill against their own cold
measurements of 448 at 40k and 253 at 161k; a `recipes/README` entry listing "vLLM at MTP=0"
among configurations they had not actually run; and a decode-table caption implying uniform
residency.

**We cite none of these.** Our references to them are the ~22 tok/s llama.cpp figure and the
inference-atlas 33.6 tok/s `serve-single-i256-o256-v1` run, neither of which was corrected.

**The asymmetry worth noting:** they now state plainly that they never ran their own MTP-off
A/B, and that the in-engine `1.6x` circulating for this model is upstream's, not theirs. We ran
that A/B on 2026-08-29 (`speculation-on-flash-next.md`):

| | c=1 decode | c=16 aggregate |
| --- | --- | --- |
| MTP off | 26.4 | 96.6 |
| MTP k=2 | 38.0 | 99.1 |
| gain | **1.44x** | **+2.6%** |

Same box, same checkpoint, same harness, both arms the same day, conditions stated
(i4000/o512, `--max-num-seqs 16`). As far as we can tell this is the only first-party MTP-off
A/B published for Flash-Next on a single GB10. It is not comparable to their 1.16x — different
engine, different quantization, different drafter — and the two should not be averaged.

Their failure modes are ours: a stale number surviving next to the data that refutes it, and a
rate extrapolated flat across a range where it is not. Both bit us this week too.

## 2026-08-29 06:19 — 0xBakeer establishes a noise floor, and it invalidates a claim of ours

"Turn WARM off by default, and record what the warm actually achieves (#3)". Residency set
deliberately and verified with `mincore(2)` before each run:

| residency at start | aggregate tok/s | per-request p50 | tpot p50 |
| --- | --- | --- | --- |
| 0.06% | 37.43 | 39.72 | 25.18 ms |
| 0.06% | 34.99 | 36.39 | 27.48 ms |
| 25.88% | 34.12 | 35.57 | 28.11 ms |

**Two identical cold runs differ by 6.5%.** The warm result sits inside that spread. Their
conclusion is the important sentence: *"nothing below roughly 10% is callable from single runs —
which is the regime both the original +42% and its retraction were working in."* Our previous
field note carries that `+42%`; it is now retracted along with its retraction.

Three further findings:

- The warmer cannot reach the 79% residency the old table claimed — from 0.06% it reads all
  26.8 GiB at 1.01 GiB/s and lands at **25.9%**, because after the model load takes its share of
  a 121 GiB box the page cache has nowhere to put the rest.
- **A warm performed before the server starts is discarded entirely**: 18% established with
  nothing mapping the file reads back as 0.06% once `llama-server` has loaded.
- **Intermediate residency is not a holdable state**: 18% before startup → 0.06%, 18% after
  startup → 11.43%, 58% → 100%, and `fadvise` cannot go below 28% while the server maps the file.
  "Two careful measurements disagreed — both were sampling a moving target."

Also, unrelated to speed and worth knowing: **vision works in the vLLM recipe and cannot work in
the GGUF one at all** — 333 vision tensors present and 0.967 on their image eval, against a GGUF
with none.

### What this costs us

We published **"k=2 is the optimum, and k=3 is already past it (36.8 against 38.0 at c=1)"**.
Our own k=2 measurements span **36.2–38.0** on identical settings, so 36.8 sits *inside* the k=2
range — the claim compares k=3 against the high end of k=2's spread, which is the same
endpoint-vs-spread error they just retracted. See `speculation-on-flash-next.md` for the
corrected wording and our own measured noise floor.

What survives is the counter-based result, which is not a timing and is far more precise:
k=3 yields **2.471 tokens per iteration against k=2's 2.133** (+15.8%) with no throughput gain.

### Reported back: 0xBakeer/qwen38-flash-next-spark#6

Posted 2026-08-29: <https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6> —
**landed and closed the same day** (commit `b80625a`, "Land the findings from #6 with
third-party attribution"), credited in their `CREDITS.md`.

Three items, scoped deliberately narrow because most of what we know is about a checkpoint we
built ourselves and nobody else has:

1. **Their 6.5% noise floor replicated at 6.9% on vLLM** — different engine, quantization and
   drafter, same hardware. Lets them state it as a property of the box rather than of llama.cpp.
2. **The MTP-off A/B their `recipes/vllm-longctx/README.md` says was never run**: 26.4 -> 35.7,
   +35% at c=1, not measurable at c=16.
3. `VLLM_TORCH_PROFILER_DIR` is inert in this build; profiling needs `--profiler-config`.

Led with the caveat that none of it is comparable to their llama.cpp numbers — they have just
spent two commits cleaning up exactly that conflation. Included our own withdrawn "k=2 is the
optimum" claim as an instance of the endpoint-versus-spread error they retracted, and flagged
that TTFT variance is uncharacterised so the -30% there is indicative only.

Deliberately **not** sent: the hyper-connection profile and the `fp8head` results. Both depend on
a local checkpoint and belong here, not in their repo.


### Outcome of #6 — how they handled it

Worth recording as a model of what to do with an unreproducible external report.

**Verified independently, adopted as theirs:** the profiler finding, all three parts — they
checked `envs.py` for the absent `VLLM_TORCH_PROFILER_DIR`, quoted the exact gate at
`entrypoints/serve/profile/api_router.py:40`, and confirmed `--profiler-config` is accepted
where `--torch-profiler-dir` is not. The systemd `Environment=` quote-stripping trap went into
their known-issues list.

**Not verified, attributed rather than adopted:** the 6.9% noise floor and the MTP-off A/B.
Their reasoning is one we should copy — our checkpoint is a local NVFP4-FP8 build with a
requantized `lm_head` that their repo cannot rerun, so the docs carry it as *our* claim on
*our* stack, and the platform-wide reading is conditional: "if their numbers hold".

They also added an endpoint-versus-spread section, calling it "the fourth way to measure this
wrong", and used **our own withdrawn k=2-optimum claim** as its worked example. The retraction
travelled further than the result did.

Lesson for our own reports: send the negative and the withdrawn alongside the positive. It was
the part they could use without rerunning anything.

## 2026-08-29 (afternoon) — the field independently reached our null, three ways

Swept the field after our own three interventions on the hyper-connections all measured null.
**Every one of our conclusions was reached independently by someone else, and one of them used our
exact shape.** That is worth more than the result itself: it means the null is a property of the
layer, not of our method.

**dolf3131 measured `(10240, 320)` at M=1 and got the same nothing** — by kernel selection rather
than precision. From `scripts/patch-skinny-gemm-tp1.py`: 1.70x at M=1 in microbenchmark, *"no
difference end to end"* (13.86 against a 13.95–14.09 band). Their two methodological findings are
worth more than the number:

- **Their microbenchmark is L2-resident and lies by ~2.5x.** Any candidate timing below
  `N*K*2 / 273 GB/s` is measuring cache. We had already built that guard into
  `tools/shapebench.py` independently.
- **A large ratio on a small weight is nothing.** Four tuning rounds won 1.3–2.3x each in
  isolation and moved end-to-end by 0.0–0.4%.
- Trap that cost them a sweep: the kernel needs `K % (block_size * vector_width) == 0`, so
  **K=320 has no valid config above `vector_width=2`** — omitting widths 1–2 makes the shape look
  unsupported.

**hn7305 quantized the hyper-connections and shipped it disabled.**
`hn7305/Qwen3.8-Flash-Next-NVFP4-Spark` implements it and records it as **measured net-negative** —
NVFP4 rather than FP8, same axis, a stronger verdict than ours. No published checkpoint quantizes
these layers: MESHIVEAI excludes them with 97 patterns, Saren with `-:.*hyper_connection.*`.

**b12x keeps the operator in BF16 on purpose.** Its `nvidia.gb10.48sm` profile pins
`norm.hyperconnection` to `dtype: bfloat16, backend: cutedsl, hidden_size: 2560, lowrank: 320` —
our exact geometry — while its GB10 `gemm.block_fp8_linear` coverage is `in_features: 2560` only.
There is no FP8 branch for K=320 anywhere in that profile. **The win they pursue is fusion, not
precision**, which is exactly where our own measurement landed.

**Upstream's fix for this operator is a better BF16 kernel, and it is gated off our hardware.**
FlashInfer PR #4266 (merged) adds a Blackwell CuTeDSL BF16 split-K GEMM: 1.463x at M=1 for
N=256/K=8192. SGLang uses it for HyperConnection Mix at M<=16 — 12.36 -> 6.03 us, +7.6% end-to-end
— but it is **SM100-only**, so sm_120 and sm_121 fall back to a persistent Triton Mix whose own
rationale is *"at these sizes every kernel is latency-bound, so the win comes from kernel count,
not bandwidth."* Note split-K is useless at K=320 regardless.

### The lever the field is actually using, and we already have it

blazux A/B'd the one that matters on a single GB10: **NVFP4 25.7 -> hybrid blockwise-fp8 30.8
tok/s (+20%)**, quality unchanged (45/51 both ways), resident 84 -> 77 GiB. That is quantizing the
~15 GiB BF16 **dense side path** — GDN `in_proj`/`out_proj`, QSA `q/k/v/o`, shared experts. It is
the highest-confidence number in the field because it is a controlled A/B rather than a headline.

**That is our `fp8head`.** Same lever, banked days ago as +39%. Notably his fp8 rewrite covers 300
tensors and **excludes the hyper-connections and `lm_head`** — the two things we went after are
the two the whole field leaves alone.

### Where we stand on speed

Nothing in 0xBakeer's atlas beats us: his best GB10 vLLM single-stream is **30.96 output / 33.43
decode-p50** against our 36.5. Four public claims exceed ours, all speculation-heavy single runs
with real caveats — hn7305 48–60 (crashes past ~130k ctx, KV pool non-deterministic), Saren ~49
(best-of-two), YSLAB 44.23 (denominator includes hidden reasoning tokens), hashd1ve 41.5 (~20-token
prompts; **27.3 at 8k, 24.7 at 128k**, i.e. below us at real context).

Two independent confirmations of our own choices: YSLAB's MTP depth sweep 0–10 finds **MTP=2 wins**
(44.23, 73.7% acceptance; acceptance collapses from depth 5), and hashd1ve finds single-stream
**latency-bound, not bandwidth-bound** (C1 42.8 / C2 53.2 / C4 95.2 aggregate).

### Leads worth checking against our own build

- **hn7305's qkv fusion scale bug.** SGLang fuses q/k/v and takes
  `alpha = input_scale.max() * weight_scale_2.max()`, silently over-dequantizing every member with
  a smaller scale. **Cosine similarity cannot detect it** — uniform magnitude error, direction
  unchanged. Cost when present: GSM8K 0.850 vs 0.965, accept length 1.013 vs 2.559. Our checks
  would not have caught this.
- **blazux's sm_121 kernel misselection** in `flash-linear-attention`: `DEFAULT = 102400` against
  GB10's 99 KiB/block, so all 36 GDN layers take small-tile kernels; `101376` gives +20% decode.
  **Checked and does not apply to us** — `fla` is not installed and vLLM reads 101376 correctly
  (verified: it agrees with Triton).

## 2026-08-29 (evening) — two more serious players, and three claims of the field's that do not hold

### styles01/sparkrun-recipes — ahead of us on serving config

<https://github.com/styles01/sparkrun-recipes> (63★, pushed daily). DGX Spark recipes with real
patch files, not just write-ups.

**Their Flash-Next vLLM runbook had two settings our launcher was missing:**
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` and `--max-model-len 262144`. We were
serving with neither. Reading their runbook is what closed both gaps.

Worth taking: **`patch_int8_lmhead_v3.py`** — INT8 W8A16 head via a batched GEMV inside
`LogitsProcessor._get_logits`, **3.35 ms vs 8.8 ms at B=1**, argmax-exact, and it *frees* the dead
BF16 weight (~1.4 GiB) into the KV pool. Their own hard-won note matches our finding from the
other side: *"the v2 B>4 loop was what made spec decode SLOWER"* — they hit the head↔speculation
interaction too. Also packages blazux's `patch_fla_shmem.py` (the 102400→101376 constant), which
we checked and ruled out for our path.

We opened <https://github.com/styles01/sparkrun-recipes/issues/2> with the noise floor, the
hyper-connection null and the lm_head×speculation numbers, plus the tool-parser measurement below.

### alesha-pro/qwen38-flash-next-4x3090 — the best validation contract in the field

<https://github.com/alesha-pro/qwen38-flash-next-4x3090>. W4A16 + FP8 PLE + **calibrated FP8 QSA
KV** on 4×3090. Ampere, so W4A16 is forced and the quant choice does not transfer — but two things
do.

**The KV lever.** Their patch map is explicit about what FP8 QSA KV costs: dtype/scale plumbing, an
FP8 decode kernel, and **calibrated scales that are mandatory — "no scale=1 fallback"** — mounted
only for `KV_CACHE_DTYPE=fp8*`, with BF16 KV preserved as a rollback. Their 12 QSA layers match our
architecture exactly. This roughly doubles the KV pool and matters now that we serve 32k rather
than 8k.

**Their validation contract, worth copying verbatim:**
- verify all 12 QSA layers *log* calibrated K/V scales — no assuming the calibration loaded;
- **no fallback to scale 1.0**;
- record exact-match separately from semantic equivalence;
- **do not label top-N KLD as full-vocabulary KLD** (our divergence harness should carry this);
- *"No eager mode, no language-model-only mode and no graph disablement are valid capacity
  workarounds."*

Also a rule to add to our offline gate: **"shared-expert gate repair — build-time restore to BF16;
mandatory."** Our `shared_expert_gate` is unquantized, so we comply — by inheritance, not by check.

### Three field claims that did not survive checking

- **"GMU 0.72+ hard-freezes the Spark" — refuted, but re-scoped after a source check.** Traced
  second-hand to a 0.76 figure. The UMA-freeze runbook does record 0.88 → watchdog kill, 0.85 →
  watchdog kill, and **0.70 → full host freeze, the worst outcome of the set**, and it opens
  *"Status: diagnostic runbook. No fix is claimed."* Two corrections to what we wrote first: it is
  a **dual-node Ray + expert-parallel** startup, not a single GB10; and the mechanism it gives is
  that utilization bounds only weights+activations+KV, while the growth is in **Ray's object store
  (~30% of host memory by default), Ray GCS/dashboard, EP all-to-all buffers, and page cache from
  reading ~100 GB of shards** — none of which the ratio touches. Our earlier "allocator staircase
  in `ModelOptNvFp4FusedMoE.process_weights_after_loading`, ~110 GiB needed against a ~107 GiB
  spike" was not in the source and is withdrawn.
  The one single-node datapoint worth having is a TP=1 preset comment: **0.90 swap-thrashes during
  weight load + torch.compile; 0.80 is stable.** That is about load and compile headroom, not KV
  sizing — and it is the only part of this that bears on us, because we run **0.90 at TP=1**.
- **NVFP4 KV is closed.** Two independent GB10 measurements: 12.13 vs 19.78 tok/s, and 48.1 vs
  54.1 against fp8+MTP, plus a structural MTP-acceptance penalty, plus silent failure. Keep one thing from it: **accept-length pinned at maximum is a corruption
  signature, not health** — one case read 3.00/3 while GSM8K scored 0/10. Cheap to add to our
  harness.
- **The FlashInfer AOT prebake is unnecessary for us.** The mechanism is real (`is_aot` is
  `aot_path.exists()`; a copy to `aot_path` stops ninja forever), but `flashinfer-jit-cache`
  already ships **960 prebuilt `.so`, 2.2 GB**, and our startup shows **zero ninja invocations**.
  It is a post-driver-upgrade recovery procedure, not a fix to run now. ⚠️ FlashInfer **0.6.18
  drops SM121a cubins** from the aarch64 cu130 jit-cache wheel — do not bump that package
  casually; we are on 0.6.17.

### Settled by us: the tool-parser question the field calls contested

`qwen3_xml` vs `qwen3_coder` was open with no published tool-call accuracy either way. Measured:
**32/32** across temps 0.2 / 0.6 / 1.0 / default, correct function name every time. And the
contest is nominal — in this build `vllm/tool_parsers/__init__.py` maps both names to the same
`Qwen3EngineToolParser`.

### Verified arch facts from the patch-repo sweep (2026-08-30)

Three findings that survived a second, file-level pass and bear on our build:

- **SM120 and SM121 have 99 KiB (101,376 B) shared memory per block, not the 228 KiB "Blackwell"
  figure** — that is SM100/B200 only (CUTLASS maintainer, NVIDIA/cutlass#3144). CUTLASS's
  `StageCountAutoCarveout` assumes the larger budget, picks ~6 stages, and overflows on the
  Pingpong schedule. It surfaces as a **device-side assert at `nvfp4_blockwise_moe.cuh:78`**, which
  is the next `cudaMallocAsync` sync point and *not* the root-cause line.
  **Runtime workaround, no patch: pick the `flashinfer_cutlass` MoE backend.** `triton` and
  `cutlass` both hit the path; `flashinfer_cutlass` avoids it.
  ⚠️ **We already run it — this was written as if it were an untried lever, and it is not.**
  `moe_backend='auto'` resolves to `FLASHINFER_CUTLASS` on this box and always has:

  ```
  [nvfp4.py:291] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends:
    ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED',
     'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'HUMMING', 'EMULATION']
  ```

  The unquantized MoE of the MTP drafter resolves to `FlashInfer CUTLASS` too. So the SMEM
  overflow is a hazard we were never exposed to, which is why the `nvfp4_blockwise_moe.cuh:78`
  assert has never appeared here. The fact is still worth holding — it explains a **non-event**,
  and it would bite immediately if anyone forced `--moe-backend triton`.
- **Arch flags fail silently, three different ways, in three different repos.** sm_121-only
  `NVCC_GENCODE` (missing sm_120) makes `EFFICIENT_ATTENTION` SDPA return output **12–27× off a CPU
  reference** with no NaN, no warning — fix is `TORCH_CUDA_ARCH_LIST="12.0;12.1"`. A
  `CUDA_SUPPORTED_ARCHS` list ending at `12.0` clamps `12.1a` down before the FP4 family match.
  NVFP4's `cvt.e2m1x2` needs `sm_121a`, not plain `sm_121`. We set `CUTE_DSL_ARCH=sm_121a` and
  `TORCH_CUDA_ARCH_LIST=12.1a`; worth a look if we ever compile our own torch.
- **Provenance note:** the first pass over these repos asserted specifics for three of them without
  reading the files. Those entries were rewritten above against the sources. The lesson is the one
  already in this file — *a repo's file tree is not a finding*.

### SGLang, checked before recommending it (2026-08-30)

Written up because it reverses a recommendation made an hour earlier in conversation: SGLang looked
like the largest unexplored lever for this model (MiaAI-Lab's 64 tok/s is on it; on a *different*
model our SGLang+DSpark path beat vLLM+MTP by 12% with 28% lower TTFT). The upstream tracker says
otherwise.

- **[sgl#36558] QSA decode has no working kernel path on SM121 — "Qwen3.8-Flash-Next unservable"**,
  open since 2026-08-26. `_resolve_trtllm_sparse_decode()` rejects GB10 on an
  `is_sm100_supported()` gate; classic FA2 is absent and has no Blackwell kernels; the resolver
  falls through to the flash-attn-4 CuTe interface, which fails to compile for this GPU. Every
  launch config crashes at first decode or during decode graph capture.
  The reporter verified the flashinfer trtllm sparse decode **kernel itself runs correctly on
  SM121** (`max_abs_diff` 4.70e-04). So this is a **gate, not a kernel** — the same shape as the
  four layers of "installed but not running" we peeled back on the skinny GEMM.
- Base support ([sgl#36497] *Introduce Qwen 3.8 Flash Next*) and the SM120/121 QSA resolver fix
  ([sgl#36556]) are both **still open**. There is no released SGLang that serves this model on GB10.

**Conclusion: SGLang is not a stack we can switch to, it is a build-from-unmerged-PRs project with a
published blocker.** Demoted from "biggest lever" to "watch #36497 / #36556 / #36558".

### Two SGLang issues that do bear on our KV plan

- **[sgl#36797] NVFP4 KV regresses Qwen4Exp decode ~29% on SM121 vs fp8_e4m3.** Measured 44.0 tok/s
  (nvfp4) against **56.8–58.6 (fp8_e4m3)** and 54–59 (bf16), same weights, 2× Spark TP=2. Third
  independent confirmation that NVFP4 KV is closed — and note what it says about the alternative:
  **fp8 and bf16 are roughly speed-neutral.** The case for FP8 KV here is **pool size, i.e. context
  and concurrency headroom, not decode rate.** We should stop pitching it as a throughput lever.
- **[sgl#36545] fp8_e4m3 KV + QSA crashes**: the QSA FA4 decode call receives **BF16 queries with
  FP8 K/V** and asserts that all three dtypes match. On SM120, on the same RadixArk weights.
  This is precisely the plumbing alesha-pro's FP8-QSA-KV suggestion needs, failing in another
  stack. **Check vLLM's QSA decode dtype handling before building anything** — it is the same
  attention design, and this is a ten-minute source read against a multi-hour build.

Incidental corroboration from #36545's launch line: it runs `--fp4-gemm-backend flashinfer_cutlass`,
consistent with the SM120/121 CUTLASS SMEM-overflow workaround noted above.

## 2026-08-30 — two corrections to this file's own field table

- **llama.cpp's qwen4exp support merged** (ggml-org/llama.cpp#27742, master, 2026-08-27, merge
  `6c84c7d`), and it subsumed both of 0xBakeer's patches: master now implements `can_reuse()` on
  `llm_graph_input_qsa` and `llm_graph_input_ple` itself, and bounds quantizer staging in slabs by
  `max_buf_size` — a more general fix than the PLE-specific one. They verified this the right way,
  with `git apply --check` against master rather than by assuming.
  For us this only re-scoped a sentence at the time. ⚠️ **Both are now moot: vllm#53896 itself
  merged 2026-08-31 05:57**, so the vLLM implementation is on `main` too — and the package was
  renamed `qwen3_8_flash_next` → `qwen4_exp` before merge, which means every source path cited
  anywhere in these notes is from the pre-merge build.
- **"`--parallel 1`" in our table read as a limitation, and is not one.** 0xBakeer withdrew the
  "concurrent requests abort the server" claim on 2026-08-27 after a reader showed eight
  simultaneous requests all returning 200 — they queue, they do not crash — and has since run
  `--parallel 2` end-to-end. The single-stream figure is what their *harness* does, not a ceiling.
  Corrected in the table above.

### The MoE-backend axis, closed properly (2026-08-30)

Prompted by being asked why `flashinfer_cutlass` was untried. It wasn't — see above. Full state:

| backend | status |
| --- | --- |
| `FLASHINFER_CUTLASS` | **what AUTO picks, and what every measurement here has used** |
| `flashinfer_b12x` | rejected at the time — `not supported for unquantized MoE`. ⚠️ The reason given here (*"`--moe-backend` is global"*) is **wrong**: `SpeculativeConfig.moe_backend` sets the drafter's backend independently. The real blocker is that b12x faults with an illegal memory access on sm_121 (vllm#50189) |
| `triton`, `cutlass` | known to hit the SM120/121 CUTLASS SMEM overflow (99 KiB budget vs the 228 KiB assumption) |
| `FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL[_BATCHED]`, `VLLM_CUTLASS`, `MARLIN`, `HUMMING` | untried, **and no field evidence favours any of them on sm_121** |

Field check found nothing evaluating NVFP4 MoE backend choice on sm_121 beyond vllm#47982 (a
`flashinfer_b12x` bug at `dp_size>1`, not our configuration). So the remaining backends are cheap
to sweep but have **no prior suggesting a win** — this is a "no reason to expect anything" axis, not
a promising one, and it should not be ranked above prefill/TTFT work.
## 2026-08-30 — DJLougen/Qwen3.8-Flash-Next-One-DGX-Spark

New entrant, created 2026-08-26, one GB10, llama.cpp lane populated and **vLLM/SGLang lanes
deliberately empty**: *"SGLang and vLLM stay fail-closed until someone lands measured Spark evidence
in those directories."* That is an open invitation we can answer — we have exactly that evidence.

**Their kernel result is real work.** A 54 KB QSA patch (fused `ggml_get_rows_mean` + RMS weighting,
`__ldg` half2/float4 loads on lightning WMMA, compact FA gather at `topk=2048`, indexer Q padded
4→8 so lightning hits WMMA, PDL) roughly **doubles long-context decode**:

| ctx | unpatched | patched |
|---:|---:|---:|
| 65,536 | 11.35 | **18.73** |
| 229,859 | 5.60 | **11.55** (2.06×, TTFT flat at ~1,200 s) |

They also publish what they *reverted* — 8-warp MMVQ, 4-head lightning inner loop, dirty-block skip
for indexer K — which is the half most repos omit.

### The number that matters to us: prefill

Their cold depth curve gives **370–416 tok/s prefill** (32,627 tokens in 78.68 s ≈ 415 tok/s at
32k).

⚠️ **Correction, same day.** This entry originally continued *"our TTFT at 32k is ~100 s, i.e.
roughly 320 tok/s"* and concluded they were ahead of us on prefill. **That number is not ours.** It
was carried over from the Qwen3.8-**27B** work (`~100 s at 32k`) and applied to Flash-Next, which is
a different model on a different stack. **We have never measured Flash-Next prefill at 32k.** What
we have measured is 4000-token inputs at c=1: TTFT **1.691–1.941 s**, i.e. roughly
**2,100–2,350 tok/s** — so the claim that we were behind was manufactured out of a borrowed figure,
and is withdrawn. Depths still differ, so no ranking against their 32k number is claimed either.
Measuring our own depth curve is the way to settle it.

### Three findings that corroborate or caution ours

- **`mtp.*` tensors are BF16 even in an FP8-tagged repo** — their converter logs show all 31 as
  `torch.bfloat16`. We found the same when building our checkpoint (`mtp.*` stays in `ignore`).
  Independent confirmation of a thing that is easy to get wrong silently.
- **Acceptance is batch-size dependent, and can take the output with it.** Same binary, draft and
  prompt: at `-b 512 -ub 128` → 33.3% accept; at `-b 2048 -ub 512` → **3.7% accept and the target
  output diverges**. Different stack, but it is a direct warning about our own acceptance-gap item
  (ours 56.6% against 73.7% and now their 75.6% at n-max 3): an acceptance number is only meaningful
  next to its batch geometry.
- **Speculation loses at long context.** `draft-mtp` on their QSA tree at 229k: 10.2 tok/s at 43%
  accept, *slower than the same kernels running plain autoregressive*. And MTP costs prefill at
  depth — 8.4% and 6.7% below unpatched at 16k and 32k. Our own queued "does MTP cost throughput
  under load" question is the concurrency-axis twin of this.

### Method worth stealing

**They lock output hashes before making a speed claim** (`2689367b205c16ce` at 4k,
`8547299278d81f66` at 64k/128k), and label every row that used a different protocol as
not-comparable. That is precisely the guard that would have caught our corrupt-shard episode, where
size-correct byte-corrupt shards produced fluent garbage invariant to every config change. We
verify checksums on *weights*; they verify hashes on *outputs*. We should do both.

They also record 0xBakeer's CUDA-graph-reuse patch as **rejected — it segfaulted** — while their own
tree reuses graphs (304 at 64k, 563 at 128k, 958 at 229k) by other means.

## 2026-08-30 — spark-arena.com, and the most comparable external run yet

**A new field resource:** <https://spark-arena.com> is a DGX Spark benchmark leaderboard with
per-submission recipes and a raw CSV endpoint (`/api/benchmarks/<id>/raw`). Worth watching; the page
is a Next.js app, so the data lives in the RSC payload or that CSV, not in the rendered HTML.

**Submission `e9307821`** (Raymond, single Spark, TP=1) is the closest thing to a like-for-like
comparison we have found: **same runtime (vLLM), same hardware, same model family.**

- Model: `provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4` — **the PLE table itself in NVFP4**
- Container `ghcr.io/provsalt/qwen3.8-flash-ple-nvfp4@sha256:a357fa93…`
- `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT=900`,
  **`VLLM_PLUGINS=qwen38_nvfp4_ple`**
- `--max-model-len 262144`, `--gpu-memory-utilization 0.9`, MTP **k=3**,
  `--mm-encoder-tp-mode data`, `--reasoning-parser qwen3 --tool-call-parser qwen3_xml
  --enable-auto-tool-choice`
- Notably **no `--distributed-executor-backend mp`** and no `--max-num-batched-tokens`

| depth d4096, c=1 | theirs | ours |
| --- | ---: | ---: |
| decode tok/s (`tg128`) | 16.2 | **36.5** |
| prefill tok/s (`ctx_pp`) | 1,261 | **~2,100–2,350** (4000-tok input, TTFT 1.69–1.94 s) |

We are roughly **2.2× on decode and ~1.7–1.9× on prefill** at comparable depth. The likely reason is
the part of our stack that is not in theirs: FP8 dense projections and an FP8 `lm_head`. Their
checkpoint is the published NVFP4, which leaves the dense projections in BF16 — the exact +39% lever
from our own ladder.

**Three things to take from it anyway:**

1. **An NVFP4 PLE table exists, ships, and loads** — via a vLLM plugin (`VLLM_PLUGINS=qwen38_nvfp4_ple`)
   and a public container. NVFP4 PLE is ~26.8 GiB against the ~47.7 we run at FP8, so this is ~21 GiB
   of unified memory back. We had the size on our map but no working checkpoint; now there is one.
2. **Their prefill is remarkably flat with depth** — `ctx_pp` sits between 1,231 and 1,600 tok/s from
   d4096 all the way to d100000, while *decode* decays hard (16.2 → 1.5 at c=10/d100k). If that
   flatness is real it is a useful target shape for our own depth curve, which we have never
   measured.
3. **They run PLE offload without the `mp` executor**, with a 900 s ready timeout instead. Either
   their build carries the uniproc fix, or the timeout papers over the startup race. Worth knowing
   before we tell anyone `mp` is mandatory.

### Provenance discipline, from two directions (2026-08-30)

Two competitors independently converged on the same gap, from opposite ends:

- **0xBakeer stamps the build into the artifact.** `setup.sh` labels the image with the upstream
  repo/ref/sha and `serve.sh` prints the sha in its startup banner, because `UPSTREAM_REF` defaults
  to `main` and *"two people building a week apart get materially different servers and neither can
  tell which one they have."* Images predating the change report
  `unknown (image predates the build label)` rather than an empty string — a nice touch, since a
  blank field reads as "no drift" when it means "unknown".
- **DJLougen hashes the output before claiming a speed.** Output hashes are locked
  (`2689367b205c16ce`, `8547299278d81f66`) before any tok/s figure is quoted.

**Where we stand.** Our pinned `0.1.dev20073+g8e685d198` already embeds the git sha, and the recipe
`run.sh` refuses to start on a mismatch — so the *input* side is covered, commit-precisely. What we
do not do is record that sha beside each measurement, or hash outputs at all. Both are cheap and
both would have caught real incidents here: the corrupt-shard day (fluent garbage, invariant to
every config change) and the venv-copy shebang trap (eight measurement arms invalidated because the
binary was not the one we thought).

**And a caution we should apply to ourselves:** they report prefix-caching behaviour changing
upstream between 2026-08-26 and 08-29. That is their container repo rather than vLLM, so it does not
transfer — but *"prefix caching is inert below 1600 tokens on this model"* is a **build-scoped**
claim, and we published it today. Ours is anchored to a named sha, which is the right side of that
line, but the anchor has to stay attached to the claim.

## 2026-08-30 — prefix caching measured properly, on vLLM, by someone else

0xBakeer#18 turns prefix caching on by default after finding the stated reason for disabling it
("a GB10 GDN kernel bug") had **no source anywhere in their repo**. Measured correctness first,
then benefit, on a shared-prefix workload at c=16:

| | caching off | caching on |
|---|---:|---:|
| aggregate decode | 46.50 tok/s | **81.79** (1.76×) |
| TTFT p50 | 5.86 s | **2.55 s** |
| wall clock | 1,020.9 s | **573.7 s** |

Hit rate 66.5% over the run; `eval-format-v1` scored **30/30** with caching on, matching the
cache-free cell.

**Independent confirmation of our block-size finding.** Their hits land "on 1,600-token block
boundaries" — the same number we derived from `Setting attention block size to 1600 tokens`. Two
different setups, same boundary. It also held with the community image's `block_size` patch
**reverted** to what vLLM `main` carries, so the boundary is not an artifact of that patch.

**A caveat of theirs that we should keep applying to ourselves:** *"Every prefill figure published
in this repository was measured cache-free… the 30,728 tok/s is a cache-assisted number on a
workload built from shared prefixes — not a prefill speed."* Our own depth curve is safe here, and
checked rather than assumed: `bench_client_real.py`'s `make_prompt()` builds a unique prompt per
request (random corpus slice plus a `[req uid random]` header) expressly to bust the cache. So
`depth-curve.md` is a cache-free curve and comparable to cache-free numbers only.

### An open tension with our own determinism result

They report **three identical temperature-0 requests over an 8.6k prompt returning byte-identical
answers**, with real cache hits behind calls 2 and 3. We ran eight identical temperature-0 requests
and got **eight distinct outputs**. Same runtime family, same hardware, same 1,600-token boundary.

Not necessarily a contradiction, and worth stating before it gets read as one:

- **We run MTP k=2; their cell is speculation-free.** That is the leading candidate and is being
  measured now against an MTP-off server.
- **Generation length differs by an order of magnitude.** Their answers are short; ours ran to
  ~3,000 completion tokens (4,283–6,714 characters). Divergence probability compounds with length,
  so three short answers agreeing does not establish that three long ones would.

What our result does settle, independently of the cause: **the prefix cache is not the source.**
Requests 1 and 2 had *zero* cache hits and still differed from each other.

## 2026-08-30 — slots are not `--max-num-seqs`, and a third external prefill number

0xBakeer#19 moves their llama.cpp default from `--parallel 1` to `2`: 1.24× on a c=16 workload,
1.30× on c=8, single-stream decode unchanged, `prefill-32k` unchanged at 0.3%.

**Two things to take, and one not to.**

- **Do not read it as our SEQS result being wrong.** They spell out why the knobs differ:
  *"llama.cpp divides `--ctx-size` across slots"* — 1 slot gives 262,144 context per request, 2
  slots gives 131,072. Their parallelism is bought with context. vLLM's `--max-num-seqs` does not
  work that way: the KV pool is shared dynamically and `--max-model-len` stays per-request. So their
  1.24–1.30× and our null at 16→64 are answers to different questions, and neither transfers.
- **A third external prefill datapoint at 32k: 1,481 tok/s**, on their patched llama.cpp build. That
  is well above the 415 tok/s the other llama.cpp repo reports at the same depth, which is itself a
  useful reminder that "llama.cpp prefill on GB10" is not one number. Ours measured
  **2,368 (MTP) / 2,136 (no MTP)** on the same depth, so we remain ahead of both, by ~1.6× against
  the faster of them.
- **Method worth noting:** their first one-slot run started at **0.06%** page-cache residency
  straight after a restart, against its two-slot twin's **72.22%** — a confound the same size as the
  effect they were measuring. They caught it, repeated warm (75.26%), got 36.40 against 36.98, and
  used the repeat to bound noise as well. That is the same class of trap as our unified-memory
  contention rule: never measure while the machine is doing something else, and check residency
  rather than assuming it.

## 2026-08-30 — veloGB10's kernels: three ideas worth stealing

`sf-stav/veloGB10` is a Rust + hand-written-PTX engine for GB10 (no Python, no framework). It does
**not** support Flash-Next — no QSA, no PLE, no hyper-connections, no `qwen4` arch — and its README
lists Qwen3.5/3.6/3.8 plus Tencent Hy3. But `kernels/` is public and instructive.

⚠️ **Method note first:** GitHub's code search returns **0 hits** for `moe`, `mtp`, `dflash` and
`nvfp4` in this repo, all of which are demonstrably present (`src/dflash2/round.rs`, 407 local
matches for nvfp4). It is not indexed. A shallow clone plus `grep` disagreed with the API on every
term. **Do not use code search as evidence of absence.**

### 1. The "lossless-MTP contract" — a discipline vLLM does not have

From `gqa_attn_splitk_k8v4`:

> *SAME split structure, SAME reduction order, SAME merge — the **lossless-MTP contract** (decode ==
> verify col-0, every split produces the same fp32 score for a (kvh, pos)).*

They engineer the decode path and the speculative-verify path to produce **bit-identical** scores,
by holding the reduction order fixed across both. That is the exact failure mode we documented from
the other side: vLLM's speculation verifies K+1 positions in one forward, a different GEMM shape
than decoding one, so reduction order differs and a near-tie argmax can flip
([[temp0-not-reproducible-under-load]]). Their answer is to make the shapes agree by construction.

Two honest caveats: this is about *speculation losslessness*, not run-to-run determinism, which is a
different property; and our own divergence persists with MTP **off**, so this contract would not fix
what we measured. It is still the right idea and we have no equivalent.

The same discipline shows up in `gdn_rollback_b`: *"PURE BYTE COPY: no arithmetic, so it is
bit-identical to the dtod memcpys by construction."*

### 2. Asymmetric KV precision — int8 K, 4-bit V

`GB10_KV_K8V4=1` runs **int8 K with q4 V** (20 B K + 12 B V per 16 elements), with matching
attention kernels (`gqa_attn_splitk_k8v4`, `write_kv_b_k8v4`, `compact_kv_k8v4`). There is also
`GB10_KV_TQ=1`, a **3.5-bit TurboQuant** KV, and a `b=3` K variant.

The asymmetry is the interesting part: K feeds `QK^T` where error propagates through the softmax,
V is averaged where error partly cancels — so K gets 8 bits and V gets 4. vLLM offers no such split
for this architecture, and its QSA backend refuses anything but `auto`/`bfloat16` KV outright. Worth
holding onto: our "NVFP4 KV is closed" conclusion is about a *symmetric* 4-bit cache, and says
nothing about an asymmetric one.

### 3. Kernel-level golden validation

`--probe-tq` validates the TurboQuant kernels against reference goldens before any of it is trusted.
Combined with 0xBakeer's output hashes and DJLougen's locked hashes, that is three independent
projects validating at three different layers — kernel, output, and build provenance — while we
validate at none of them automatically.

### What Flash-Next support would actually need there

Present already: **GDN** (`gdn_chunk_prefill_b`, `gdn_prep_b`, `delta_step`, `conv1d_*`), **MoE**
(`moe_router_topk_sigmoid_b`, `moe_experts_fp4_b`, grouped/folded combines), **NVFP4 GEMM**,
**DFlash2 + speculation**, **prefix cache**. Missing: **QSA**, **PLE + host offload**,
**hyper-connections**. So it is decomposable rather than a rewrite — but the PLE is a subsystem
(51.2 B params in host RAM, per-token gather) rather than a kernel, and their `src/dsv4_cpu.rs`
shows the house pattern is CPU reference first, then kernel, validated against it.

## 2026-08-30 — a third KV group, and a caveat on our own prefix-cache claim

0xBakeer#20 retracts their own earlier explanation and supplies the piece we both missed. The page
alignment we observed (`Setting attention block size to 1600 tokens…`) covers **attention and
Mamba only**. There is a **third** KV group: the QSA raw-key ring, a `CircularBufferSpec` whose
block *is* its ring capacity.

**Verified in our own tree, not taken on report** (`models/qwen3_8_flash_next/common/qsa_cache.py`,
confirmed byte-identical to its `.pre-fuseddraft` backup, so we are on stock code):

```python
span     = self.compress_ratio + vllm_config.num_speculative_tokens
capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
assert self.cache_config.block_size % capacity == 0
```

Our `indexer_compress_ratio = 4`, MTP k=2 → **capacity 8**. And `v1/engine/core.py:321` sets
`cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups)` over **all**
groups — `generate_scheduler_kv_cache_config` only flattens `UniformTypeKVCacheSpecs`, it does not
drop the ring. So after that line `cache_config.block_size` is plausibly **8**, not 1,600.

### What this does and does not change for us

- **Our measurements stand.** `prefix_cache_hits_total` moving in units of 1,600, zero hits on a
  ~1,400-token prompt, hits from the third repeat on ~5,700 — that is the *attention group's* block
  and is unaffected.
- **Our wording needs a caveat.** "vLLM raises the attention block size to 1600" is right; treating
  1,600 as *the* block size is not. There are three groups, and `cache_config.block_size` is the
  minimum over them — a different number, and the one a caller gets when they ask the config.
- **We may be exposed to the split bug after all**, and our tests could not have seen it.
  0xBakeer's argument is the useful half: with the mismatch, a cold request rarely ends a chunk on a
  1,600 boundary, so it publishes no Mamba block; the repeat takes an attention-only hit and
  recomputes recurrent state from scratch — correct output, no guard hit. Reaching the zero-state
  restore needs a Mamba block published first, which is **scheduling-dependent**. So "identical
  outputs across N repeats" was never going to clear this, theirs or ours.

**Their retraction is the model to copy.** #16 concluded "nothing depends on that patch" from three
identical calls; #20 withdraws it after reading the class out of their own image. The failure was
using output identity as the observable for a bug that does not change output on the path the test
takes.


## 2026-08-30 — "a release is a measurement epoch, not an API contract"

0xBakeer#21 versions their recipes on a rule worth copying verbatim: **MAJOR** = a recipe added or
removed, or the measurement basis changes; **MINOR** = a shipped default changes, *your numbers
move*; **PATCH** = docs and corrections, numbers do not move. Every changelog entry leads with a
**"Defaults that changed"** table — was, now, and what it costs. Their trigger was two defaults
flipping in one day with nothing in the repo letting a reader tell which configuration a published
figure belonged to.

**This is our problem too, and we hit it twice today.** The `266.8 tok/s` headline belonged to the
baseline checkpoint, not the shipped one. A prefill figure was borrowed from the 27B and applied to
Flash-Next. Both are the same failure: *a number outliving the configuration it was measured on.*

And our defaults have moved repeatedly — MTP k=2 confirmed as the optimum, `--max-num-seqs` shown to
be non-binding, `--max-model-len` guidance raised 8192 → 32768, prefix caching understood as
inert below the attention block, and `4e8b849b8d97` hand-applied to the serving venv. Anyone reading
a figure in this repo has no marker telling them which of those were in force.

With build-sha stamping (0xBakeer#17) and output hashing (DJLougen), that is **three independent
provenance mechanisms** the field runs and we do not: what built it, what it produced, and which
defaults were in force. The cheapest of the three for us is this one — it is a `CHANGELOG.md` and a
rule, not tooling.

**Follow-up (0xBakeer#23): the exposure statement now includes us, and `main` after the merge.**
Their #20 said "upstream vLLM is not affected — `CircularBufferSpec` does not exist there", true of
`main` and misleading about vLLM. The class is added by **vllm#53896 itself**, to
`vllm/v1/kv_cache_interface.py` — a core file, not a model-local one — and the same PR touches both
consumers, `v1/core/sched/scheduler.py` and `v1/worker/gpu/model_states/mamba_hybrid.py`. So the
small KV group and the code that mishandles it arrive together, and reach `main` when #53896 merges.
**Our build is #53896-based**, which is why our own tree has the class and resolves capacity 8 — so
this is our exposure too, not someone else's. They are filing on that branch.

Worth noting as a pattern: this is the **third successive narrowing of the same claim** in two days
— "a GB10 GDN kernel bug" → "the alignment means the mismatch never arises" → "upstream is not
affected" — each retracted by its own author after checking. The claims got smaller and truer every
time, which is what a repo that publishes its reasoning looks like from outside.

## 2026-08-31 — MiaAI-Lab documents the empty-content trap, with a better diagnostic than ours

Their README now carries the caveat we hit twice in one session:

> *"this is a reasoning model and thinking consumes `max_tokens` first — if `content` is empty with
> `finish_reason: "length"`, just raise `max_tokens` (2–4k covers most image QA);
> `chat_template_kwargs.thinking_budget` is **not** honored by this build."*

**They give the direct signal and we did not have one.** We detected empty content by counting
characters *after* the run, which let a determinism probe report five empty strings as "identical"
twice. `finish_reason: "length"` distinguishes *"the model finished and said little"* from *"the
budget ran out inside `<think>`"* at the moment it happens — a one-field check that turns a vacuous
comparison into a loud one. Adopt it in every probe.

Also worth having: **`thinking_budget` is not honored** in this build, so the only working lever is
`reasoning_effort` (`low` / `medium` / `high`), which is what we have been using.

- 2026-09-03: llama.cpp PR #28136 reads the PLE table with parallel `pread()` instead of mmap faults — DGX Spark cold prefill 300 → 750–800 tok/s with the table on SSD. Our vLLM PLE offload keeps the table in RAM and prefills ~2,400–2,600 tok/s; the transferable idea is an SSD-backed gather to free ~50 GB of unified memory. Details in `ple-access-pattern.md`.

## 2026-09-03 sweep — GB10 prefill is the same everywhere (~2.3–2.7k tok/s cold)

| repo | stack | cold prefill | note |
| --- | --- | --- | --- |
| airawatraj/dgx-spark-qwen38-flash-agent | SGLang + **HashK GPU-resident PLE** (12.8 GB, `build_hashk.sh` ~6 min on GPU) + NEXTN | 2,406–2,500 tok/s | warm radix-cache TTFT 1.1–1.5 s multi-turn; decode 36.8 (code) / 22 (chat) tok/s c=1 |
| crimsonjoo/DGXspark1-Qwen3.8-Flash-Next | vLLM, PLE from NVMe via mmap, prefix-cache fix, **exact top-k default** (k3dani's overlay, #51782) | ~2,400 stock kernel, **1,500–2,000 with their exact top-k** (−17…−40 %; ours costs +6 %, PR #55122 ≈ +2 %) | "hybrid" NVFP4 experts + fp8 side layers: +20 % decode — same idea as our FP8-mixed checkpoint (+39 %) |
| dolf3131/qwen3.8-flash-next-dgx-spark | vLLM, PLE paged to swap | 2,719 tok/s @30k | decode 32.7 tok/s; prefix cache "needs two flags" |
| ryangu00/dell-pro-max-gb10 | vLLM TP2 dual GB10 | 2,321 tok/s @30k | worker dies from ~95k ctx on sm_121 dual-node |
| ajeetcoolkarni/…-production | SGLang, 1 GB10 | 513 tok/s **with a concurrent decoder** | prefill starves decode: 21.6 tok/s during a 32k prefill |
| bemlerlabs/…-sglang | SGLang, NVMe PLE mmap, "Triton FP4 prefill kernels" | not stated | NEXTN 3/1/4; claims 110–152 tok/s (unqualified) |
| thadreber-web/llama.cpp-qwen38-flash-next | llama.cpp GB10 | `-b 8192 -ub 8192` 9.3 % faster e2e at 35k | CUDA-graph cache fix +12–14 % gen; on-disk PLE |
| mratsim/sglang-qwen38fn-sm120-turbo | SGLang, 1× RTX Pro 6000 (sm_120, +3000 MT/s mem OC) | **11–13k tok/s** | ~4–5× GB10 — consistent with a compute-bound prefill |
| cglab-public/dgx-spark-flashnext | SGLang TP2 → **migrated to vLLM 2026-09-01** | — | SGLang sm_121 sparse-decode kernel unobtainable / unmerged (sgl#36497) |
| WeZZard/dgx-spark-bench | bench lab, 2 nodes, llama.cpp/SGLang/vLLM adapters, KL gate | — | SLA TTFT ≤ 2 s |
| official `vllm-project/recipes` Qwen3.8-Flash-Next.yaml (08-26) | docker image only, `--no-enable-flashinfer-autotune`, MTP n=3, default variant Inferact NVFP4 (PLE in nvfp4) | — | GB10 not in the verified list |

Reading: cold prefill on one GB10 is 2.3–2.7k tok/s on vLLM and SGLang alike, with the PLE on CPU,
on NVMe, in swap, or **GPU-resident (HashK)** — PLE placement does not move prefill. The one
stack-independent multiplier is the warm prefix cache (everyone reports ~10× on repeated prefixes).

## 2026-09-04 sweep — the field went after the PLE table, and one single-Spark kit claims 44 tok/s

**Checkpoints / kits**
- **myllmbox `Qwen3.8-Flash-Next-hibrid46`** + kit `bilikaz/qwen38-flash-next-recipe` (09-04, 91.2 GiB, 3 downloads):
  single Spark, vLLM (vendor SM121 image + one `ple_layer.py` patch), MTP K=3. Precision map: routed experts
  NVFP4 (Inferact), **GDN in/out NVFP4**, QSA q/k/v/o **bf16**, shared expert / lm_head / embeddings / MTP head
  bf16, **PLE table int3 asymmetric per-row (group 160) shipped as ordinary tensors** (`qbits_packed/_scales/_mins`,
  ~19 GB, declared in `config.json` `ple_quantization`) and loaded *resident*, dequant on gather — no offload
  worker, no swap, no side files. Claimed: 44 tok/s sustained single-stream (55 peak, MTP acceptance
  "3.1–3.3"), 148–158 aggregate at c=8, 19 G KV pool, 262k context, `async-scheduling` +9–12 %. Versus ours
  (finding: 69 % of single-stream is the BF16 GEMV on the unquantized dense weights + the PLE IPC): they attack
  exactly those two — the GDN projections quantized and the table on the GPU side. Quality claim: "A/B'd on
  paired outputs", no numbers. Patch upstreaming "planned".
- **primitive-ai `Qwen3.8-Flash-Next-PLE-quant`** (105 GB repo): the BF16 table re-quantized three ways —
  FP8 per-row 49 GB, INT4 g16 32 GB, NVFP4-style e2m1 g16 28.8 GB — plus a two-file overlay
  (`worker_image_quant.py`, `ple_layer_quant.py`) for the `vllm/vllm-openai:qwen38-flash-next` image that
  serves them **memory-mapped from disk** (`VLLM_PLE_QUANT_DIR`): host cost becomes page cache. Works with any
  checkpoint that keeps the BF16 tables (ours does). `huginnfork/…-noPLE` is the primitive-ai mixed build minus the
  43 BF16 PLE shards (95.4 GiB saved on disk) for use with those sidecars.
- Also new: `arnomatic/…-W4A16-PLE8` (INT4 + W8A16 g32 PLE, 2× Strix Halo), JANGQ-AI/OsaurusAI `JANG_*` bundles
  (vMLX/Mac, "median KL 0.0042 vs bf16 at 96 GiB"), `p0ly31 … MTPLX` (Mac, SSD-streamed table), `Mia-AiLab/…-NVFP4`
  (a mirror of local-inference-lab's), `WeZZard/dgx-spark-bench` (2-node, served-rate methodology, three MoEs),
  `bjk110/spark_vllm_docker` (★57, single/dual presets + patch ledger), `Dyluhn/R9V` (RDNA4 shape-specialised
  engines, one model per profile — same philosophy as antirez's DS4).

**Upstream**
- **#54882 merged 09-03 17:09**: FP8 PLE loading in mixed ModelOpt checkpoints (in `qwen4_exp/nvidia/ple_layer.py`)
  — this is our `tools/main/ple_gate_patch.py`; obsolete on the next main rebuild (our venv predates the merge).
- #54129 (Trosfy, mmap PLE, `VLLM_PLE_MMAP`) still open, 24 comments, active 09-03. #53899 (offload worker) open.
- #54928 update 09-04 16:45: Windless84 (sm_120, stock 0.28.0, 27B-FP8) shows E == V != A at the first divergence — the
  verifier's block forward ranks differently than the single-token forward; eager does not help; GDN refuses batch-invariant
  mode. Finding 109; GEMM M-invariance microtest queued (`gemminv`).
- Issues: #54928 "DFlash2 changes greedy Qwen3.8 output at token 30, incl. K=1 and eager" (BF16, 4×24 GB; one
  reply offering to trace) — same shape as our "diverges from ~30 tokens" findings, but on a BF16 27B, so not the
  NVFP4 finalize; our per-position acceptance and logprob-divergence tools would apply. #54919 long-prefill
  starving decode for minutes on 2-node TP2 (scheduler, not ours). #55279 DFlash2 cumulative OOB on sm_80.

**What this changes for us**
1. The PLE memory shape has two ready alternatives to our swap-backed offload worker: primitive-ai's mmap'd
   INT4/NVFP4 sidecars (drop-in, our checkpoint, ~30 GB page cache instead of 48 GiB anon + 64 GiB swap) and the
   hibrid46 resident-int3 route (needs their checkpoint). Either would also free ~20–40 GiB for KV.
2. hibrid46's 44 tok/s single-stream, if it holds, is above our best (FP8-mix + MTP). The two levers are ones we
   identified but did not build (quantize the GDN projections; keep the table on the GPU side). A/B needs their
   91 GiB checkpoint (download only on the user's go) and our BF16-divergence tooling for the quality side.
3. Nothing new on prefill/TTFT anywhere in the field; our L2 line remains ours.

## 2026-09-04 — MiaAI-Lab single-Spark kit (vLLM, NVFP4 + MXFP8, mmap PLE)

https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (49 stars, pushed today). Same preview image as our
finding-69/74 measurements, `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` (99 GB; MXFP8 dense via FlashInfer `mm_mxfp8` with a
BF16-emulation fallback for `N % 32 != 0` — their dimension table for sm_121 is a real finding), PLE as a packed 27 GB
file under `MADV_RANDOM` mmap (page cache, no swapfile), MTP 3, fp8 KV via their own kernel patch, compile mode 0,
batch 2048, `MAX_NUM_SEQS=4`, cgroup cap + memwatch at a 6 GiB floor. Their numbers: 8k TTFT 5.00 s (1,646 tok/s),
32k 15.83 s; decode 36.9 tok/s single-stream, 85.9 aggregate at 4. Ours on the same image: 2.7–2.8 s / 10.6–10.9 s;
decode equal. Posted issue #4 with the three transferable items (concurrency ceiling, prefill config, warm turns);
the union kernel is deliberately not offered there — their batches mix requests (our gate needs single-request
chunks) and the honest route is upstream.

## 2026-09-04 late — sweep: who else is doing real work on Flash-Next

- **mratsim/sglang-qwen38fn-sm120-turbo** (2026-09-02, ★5): SGLang on one RTX Pro 6000 (CC 12.0, 96 GiB), RadixArk NVFP4;
  **11–13k tok/s prefill, ~200 tok/s per stream, 1,170 tok/s aggregate at 8**. Seven patches on the day-0 image: fp8 KV
  on sm_120, linear-attention layers do not cache MTP drafts (−2 GB, "surprisingly not slower"), **MXFP8 at load for
  everything the checkpoint left in bf16** (attention, MLP, lm_head, hyper-connection mix), richer-calibration NVFP4 from
  local-inference-lab, Triton preload via long-prefill warmup. mratsim is a serious kernel engineer; the natural SM120
  tester for the tile-union override once the PR is up. Lists four other SM120 SGLang forks (jpezzulli, gabrielolympie,
  lovedheart, ormandj).
- **carloslfu/slotstream** (★287): Macs, experts streamed from SSD through a fixed slot pool, 12 tok/s warm decode in
  32 GB on a 48 GB M5 Pro. Memory design, not kernels; the most-starred Flash-Next repo.
- **alexskinner/qwen38-flash-spark-blend** (2026-09-04): blazux + Mia's `MADV_RANDOM` + MTP=3 on one GB10, one harness
  head-to-head: 30.4 tok/s decode, **1,863 tok/s prefill at 40k (+26 % over blazux)**, 92 tok/s at 8 streams, 1.12 s
  prefix-hit TTFT. Confirms Mia's mmap advice and MTP=3 (2.45 accepted/step vs 2.10 at MTP=2).
- **davetha/r9700-lru-expert-cache** (AMD, 2× R9700): device-side LRU expert cache decided on the GPU between decode
  steps; PCIe expert traffic 432 MB/step → much less, prose 76.7 → 91.8 tok/s with MTP-4. A real systems idea for any
  box whose experts do not fit.
- **llama.cpp**: coder543 #28136 *direct reads for the lazy PLE table* — real-task prefill 300 → 750–800 tok/s on GB10,
  root cause mmap over-read ("it's always mmap"); abdel-darwish-27 #28213 gather-based sparse attention for QSA decode;
  fairydreaming #28330 no V cache for the indexer. The llama.cpp lane is now touching the same two levers we did
  (PLE placement, QSA gather).
- **vLLM**: gau-nernst #55272 remove torch.compile for the NVIDIA implementation (open — bears on our compile-freeze
  findings); peakcrosser7 #55375 fused PLE conv state-index strides fix; aoshen02 #55341 warm up kernels before CUDA
  graph capture. **SGLang** #37995 (Jiminator) cookbook for NVFP4 TP=2 on 2× Spark with PLE offload *off* on unified memory.
- HF today: primitive-ai NVFP4 at 12k downloads, Baekpica SSD-PLE GGUF 14k, arnomatic W4A16-PLE8, tcclaviger MXFP4-FP8,
  lychee888 NVFP4-FP8PLE.
- **Mia issue #4 reply (malvavisc0, 21:02)**: deployed the kit (512k YaRN, fp8 KV, MTP 3); confirms the MAX_NUM_SEQS trap
  (blazux measured the same flatline: 33 tok/s at 2 seqs vs 267 at 48); **found that `start.sh` passes
  `--compilation-config`, `--hf-overrides` and `--speculative-config` wrapped in literal single quotes**, so the compile
  mode may never be applied — our point 2's compile row is unmeasured on their kit; warns that prefix caching on the Mia
  kit is unsafe (engine block = 8-token QSA ring instead of the 1,600-token Mamba block → zeroed Mamba state on hits;
  blazux has the two-file fix). Offers to run the batch/compile A/B once the quoting is fixed. Reply drafted.

## 2026-09-05 — `nvidia/Qwen3.8-Flash-Next-NVFP4` (official NVIDIA release)

Created 09-02, published 08-31 per the card, 123.6 GiB: 10 main shards (73.6 GiB) + one 50 GiB `model-fp8-mtp-ple.safetensors`.
ModelOpt v0.46 MIXED_PRECISION: **routed experts W4A4 NVFP4 (group 16, MSE-calibrated weight scales, calibration =
cnn_dailymail + Nemotron-Post-Training-Dataset-v2), everything else in the main model BF16** (attention, GDN, shared
experts, hyper-connections, lm_head all in `exclude_modules`), MTP routed experts FP8 128×128 block-scaled, PLE per-tensor
FP8; MTP + PLE byte-identical to `Qwen/Qwen3.8-Flash-Next-FP8`. Needs vLLM ≥ d4d703ca (= #54882, the mixed-ModelOpt FP8 PLE
fix — in our dev401). Sample command is TP8. Their accuracy table (FP8 → NVFP4): GPQA 92.0 → 91.5, HLE 34.7 → 35.4,
τ²-Telecom 90.8 → 90.1, MMMU Pro 77.1 → 78.3, SciCode 16.3 → 18.8, AA-LCR 71.9 → 74.1, IFBench 80.5 → 81.0, Omniscience
28.1 → 27.6, Terminal-Bench 2.1 83.3 → 82.9 — within noise of FP8 on all nine.

Against our stack: it is RadixArk's shape (NVFP4 experts + FP8 PLE) with **BF16 dense layers**, i.e. without our +39 %
FP8-dense lever and the FP8 lm_head — so as shipped it decodes like the RadixArk baseline (~17–24 tok/s c=1), not like
fp8head. What it may bring: better-calibrated expert scales (MSE + a richer set; primitive-ai/local-inference-lab argued
the same) → a quality comparison by our logprob-divergence method, and an official MTP module. To use it at our speed we
would apply the FP8-dense conversion on top. Disk: 60 GB free; the main shards alone are 73.6 GiB, the PLE/MTP file 50 GiB
(ours is RadixArk's FP8 PLE — identity to Qwen FP8's not verified). **Not downloaded; needs the go and ~80–130 GB freed.**


## 2026-09-08 sweep — the field is now working our lever list, and one entry validates our kernel

Read for the agent-turn goal, not for deployment recipes. Commits since 2026-09-05 on the ten repos
tracked above; four repos moved, six were silent.

**[MiaAI-Lab single-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)** is the
one worth reading in full — five measured changes, and three of them are items on our own list:

| their change | their number | our state |
| --- | --- | --- |
| `mamba_ssm_cache_dtype=bfloat16` | +6.8 % decode at 1 stream, **+8.5 % at 8**; attention block 3,200 → 1,664; needles 15/15 | **the finding-141 lever, by a route we had written off.** Finding 142 says "MambaDType has no fp8, so [the padding overlay] is the only in-config route" — but `FUSED_GDN_STATE_DTYPES = (float32, bfloat16)`, and the checkpoint ships float32. Untested here. Running tonight (`ssm`). |
| draft-vocabulary slice, 65,536 rows | **−16.9 % single-stream step**, −6.1 % at 8; MGSM en 94.8 vs 93.6 full, zh identical | we measured +6.4–6.8 % at c=1 and picked 32k (det-135); their 65k argument is better than our 32k one — the crossover where lost acceptance eats the byte saving is at 88–90 % corpus coverage, and 65k buys 3.6 points of coverage for 3 points of byte saving. Their corpus recipe (513 MiB wikitext-103 + 47 MiB Python + model output) is the part we lack: fitting the vocabulary to the model's own output does not work (52 generations = 4,250 distinct ids). |
| `posix_fadvise(WILLNEED)` before the PLE `index_select` | 280-row cold gather 20.12 ms → 1.51 ms; **−3.2 % mean step, 6/6 levels** | **does not transfer.** Their table is mmap-backed, so the gather is a fault loop; ours is RAM-resident through the offload worker and the profile puts it at 0.0 % of prefill (finding 65) with the decode handshake fully hidden (gap 0.00 ms, `decode-c1-idle-piecewise`). Their number is a measurement of mmap, not of PLE. |
| capture every `(1+K)·S` cudagraph width | ~4–5 ms on the 5-sequence step, nothing at 1/2/4 | this is what MiaAI **#19** asked us for. It also explains why our `cgsize`/`cgnone` arms were null: we only ever measured c ∈ {1,4,16}, and 4 and 16 are captured widths in both arms. The gap is at the widths *between* the captured ones. |
| `VLLM_USE_V2_MODEL_RUNNER=1` | forced; the spec draft-config copy falls back to V1 and mutates the `compilation_config` it shares with the target, silently downgrading FULL_DECODE_ONLY → PIECEWISE (+24 % single-stream step for them) | **the mechanism is real, the consequence is not ours**: det-136 A/B'd PIECEWISE vs FULL_AND_PIECEWISE vs FULL_DECODE_ONLY on this box and all three were equal. Worth one grep of our logs for "Overriding cudagraph_mode" and nothing more. |

Two corrections of theirs are worth keeping because they cost us nothing to learn:
`index_share_for_mtp_iteration` **cannot be set from the command line** — `--hf-overrides` puts it on the
*target* `text_config`, the drafter reads it from the *draft* config, and
`SpeculativeConfig.compose_draft_hf_overrides` does not propagate dict overrides
(`config/speculative.py:734`, their verification in-container). The remaining routes are a checkpoint
`config.json` edit or a patch. And `flashinfer_b12x`'s exclusion from the auto MoE backend list is *not*
stale: it selects for both processes and then faults with an illegal memory access during `profile_run`,
which matches our own veto in `moe-backend-axis.md` but attributes it to a different place.

**[blazux](https://github.com/blazux/qwen3.8-Flash-DGX)** — "patch 8: bump the deterministic top-k pin
(correctness fix + the kernel got much faster)", 2026-09-07. That pin is our PR #55122; the speed-up is
the blocked 4-item emission and the 22,016 `RADIX_THRESHOLD` from 09-08. First outside confirmation that
the kernel is being run by someone else on their own hardware. They also added multi-client/multi-agent
guidance (their issues #9/#10), which is the same workload class as our agent-turn goal.

**[veloGB10](https://github.com/sf-stav/veloGB10)** shipped "FP8 prefill levers on" and a DFlash2 tree
mode; **[styles01](https://github.com/styles01/sparkrun-recipes)** pinned MiaAI forward (+26 % decode) and
is now mostly non-LLM lanes. 0xBakeer, DJLougen, alexskinner, mratsim and cglab-public were silent.

**Order this sweep implies for the agent-turn work:** (1) bf16 recurrent state — it is the finding-141
lever and it moves decode as well as the block size; (2) the draft-vocabulary slice at 65k with a real
corpus, since it is the only change either project has found that moves the single-stream step at all;
(3) the cudagraph widths *between* the captured sizes, which also discharges MiaAI #19. The PLE prefetch
is not a lever here and should be struck from the plan.

### Open issues worth acting on (same sweep, 2026-09-08)

Four upstream threads sit on the agent-turn path. Two of them are the same coupling seen from
different sides, and we hold measurements neither reporter has.

- **vllm#54458** (open, GLM-5.3-Flash on 2× RTX PRO 6000): attention blocks inflated to **7,808
  tokens** by the same `may_override_attention_block_size` rule that gives us 1,600; ~23 KV groups ×
  one page each pins 118k–150k token-equivalents *per request whatever its length*, so only 2–3
  requests run and a cached 35k prompt is fully evicted after three interleaved admissions. Their ask
  is verbatim finding 141's lever 2 — "allow one mamba state page to span multiple attention-sized
  pool pages". **We are already its second commenter (2026-08-30).** The line in
  `warm-turn-block-granularity` claiming no open issue existed was a search failure and is corrected.
- **vllm#55533** (open, active, Qwen3.8-27B-class hybrid GDN + MTP): the *concurrency* face of the
  same coupling. `MambaSpec.num_speculative_blocks = num_speculative_tokens`, and align-mode
  `MambaManager` charges **1 + k** blocks per Mamba group for the request's lifetime, so
  `max_concurrency = floor((num_gpu_blocks − 1) / (1 + G·(1+k)))` — at batch 8 the scheduler runs
  exactly 3 sequences and MTP becomes 2–3× **slower** than no-spec. Our pool is 74 blocks of 1,600
  tokens with k=3, which is the same arithmetic. Measured tonight in `mtp4` (`schedwidth.py` polls
  `num_requests_running` at c=8, with a no-spec arm as the comparison). Both halves must hold before
  we call it a reproduction.
- **vllm#43258** (open, 09-04) emits prefix-cache KV events at hash-block granularity for hybrids and
  **vllm#51769** (open) warns when EAGLE/MTP speculation costs a large prefix-cache hit — our
  finding, upstream, as a warning rather than a fix. Neither shrinks the block.
- **vllm#53239**: `bfloat16` Mamba cache crashes `selective_scan_fwd` on an L40S. Read before
  tonight's run and set aside: that is Mamba2's SSD kernel, not GDN's `chunk_gated_delta_rule`, whose
  `FUSED_GDN_STATE_DTYPES` names bfloat16 explicitly — and #55533's reporter runs a hybrid GDN model
  with `mamba_ssm_cache_dtype="bfloat16"` without comment.

On the field's own trackers: **blazux #9** (closed) — "30 tok/s → 0.2 and back" with two clients — is
diagnosable from the log the reporter pasted, and it is not the model: `PLE mmap stats` shows
**1,518 ms per gather op** during the collapse against 54 ms/op when it recovers, gathering ~356k rows
each time. That is their mmap PLE thrashing under two clients, i.e. the same subsystem their
`posix_fadvise` patch addresses, and another reason the fadvise result does not transfer to a
RAM-resident table. **MiaAI single #28** (greedy non-determinism + context recitation inflating MTP
acceptance to 0.93 on long prose) we have already answered with the three kernel defects; the
methodological half is worth keeping for our own numbers — accept-length pinned at the maximum is a
recital signature, not health.

**MiaAI dual #28** is a config-only tuning report on a stock dual-Spark pair and contains the one
untested cell on our own list: **MTP `num_speculative_tokens: 4`**, +11.4 % over 3, with the k-curve
monotonic 0→4 (25.9 / 43.4 / 52.6 / 59.1 / 65.8 tok/s). Our sweep stopped at 3 because k=5 hard-fails
on the block-size hole, which made 5 look like a wall rather than a gap. Also there: CPU C-states off
is worth **+5–6.6 %** by live ablation (`cpupower idle-set -D 0`, pure step latency, acceptance
unchanged) — a host lever we have never tested and which needs root; and `index_share_for_mtp_iteration`
lifts acceptance 2.42 → 2.52 per draft, which pairs with MiaAI-single's finding that the flag cannot
be set from the command line at all.

## 2026-09-09 — upstream issue sweep: four threads that touch our surface, none of which we are in

**vllm#55922 — "Qwen3.8-Flash-Next optimizations tracking issue"** (ZJY0516, 09-08). A tracking index for our
exact model: kernels (#55272 remove torch.compile, #54517 fuse PLE, #54513 split QSA prefill/decode, #54560
Hopper LL-GEMM), FP8 cache (#55557 main KV on the QSA path, #54890 indexer cache), PLE/Engram offload (#54371),
MoE (#55867), PD, CPU offload. **Determinism appears nowhere in it**, and neither does anything from a
single-GB10 deployment. We hold the largest body of Flash-Next determinism and agent-turn data anywhere and are
not visible on the one page someone would look at first. Same author as #38315, where we commented today.

**vllm#55951 — the same bug class as #55375, on a different path** (09-08). "mamba align postprocess indexes
block table by **batch row** while the `idx_mapping` it is given belongs to the batch from `pp_size` steps ago"
→ recurrent state copied into the wrong physical mamba blocks, possibly a live neighbour's or a freed-and-
reassigned one; NaN logits or silent recall loss. Gated on **PP>1**, which we do not run (TP=1), so it does not
explain our residual concurrent perturbation. But it is the **second confirmed cross-request state-contamination
defect on hybrid + spec decode in a week**, after #55375. The class is live on this architecture, which is
context for det-189: we found no leakage at top-5 resolution on *our* configuration, not that the class is
absent.

**vllm#55697 — RFC: Application-Directed Prefix Checkpoints for Mamba / Hybrid Prefix Caching** (09-07). An
input-side marker (`<|mamba_checkpoint|>`) letting an application declare semantic prefix boundaries, with
producer/consumer prefill scheduling; claims **+110 % throughput (5.9 → 12.4 QPS)** on Qwen3.5-35B / L40S at
exact logit parity (< 1.2e-5). This attacks the cost finding 141 measured — redundant prefix recomputation on
hybrids — from the *application* side, where #54458 attacks it from the allocator side. We have the numbers
that page lacks for an agent workload: 1,026 recomputed tokens against 242 new per warm turn at block 1,600,
and the bf16-SSM result (block → 832, −9.6 % paired agent-turn TTFT, finding 153).

**vllm#54458 / #55533** — the two faces of the mamba/attention page coupling, both still open, #54458 untouched
since our comment on 08-30. det-188 has since narrowed the concurrency side considerably (peaked at the
incommensurate chunk budget, nested affected sets).

**Our PRs:** #55122 CI shows one FAILURE — `pre-run-check` → *"Check PR label and author merge count"*. The PR
carries **no labels**; that is a maintainer action, not a code defect, and @LucasWilkinson was already pinged
(posting log, comment `5570822909`). #55430 is still a draft, #54948 and #54912 untouched since early September.

## 2026-09-09 — the NVFP4 field for Flash-Next: the model got decomposed into body + PLE

100 repos now match `Qwen3.8-Flash-Next`; 12 are NVFP4 and **11 are about the PLE**, a category that did
not exist when we last swept on 09-05. Two things changed, and both bear on us.

**1. Mixed NVFP4/FP8 is now the consensus recipe — and it is ours.**
`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` (09-06, 5,165 downloads) is FP8 W8 on
`self_attn.{q,k,v,o}_proj` **and** `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, NVFP4 group-16 on the
MoE experts, with `ple`, `visual`, `embed_tokens`, `mtp`, `norm` and `mlp.gate` ignored. That is the same
split NVIDIA shipped for the 27B two days earlier, and the same lever as our `fp8-mixed-checkpoint`
(+39 % decode). One difference worth noting: neither of them quantises `lm_head` here, while NVIDIA's
**27B does** put `lm_head` in NVFP4 — the choice we measured as 2.4 % worse NLL.

**2. The model is now shipped as body + PLE, separately, and the PLE has been quantised hard.**

| artifact | size | what |
| --- | --- | --- |
| `primitive-ai/…-mixed-NVFP4-FP8` | 57.4 GiB | body only, PLE ignored, 47 files |
| `primitive-ai/…-NVFP4` | 55.4 GiB | body only, 0 PLE files, 14,281 downloads |
| **`primitive-ai/…-PLE-quant`** | **48.9 / 29.8 / 26.8 GiB** | the same table in **FP8 per-row**, **INT4 g16**, **NVFP4-style g16 e2m1** — pick one |
| ours (`qwen38-flash-next-nvfp4`) | **126 GiB** | body + FP8 PLE in one tree (`model-plefp8-*`, **48 GiB**) |

**CORRECTION (same day).** I first recorded this repo as "14.9 GiB, 41 files" and concluded ~54 GiB back.
Wrong: the HF tree API pages at 50 entries and I read one page of a 387-file repo as the whole thing. The
real figures are three alternative tables of **48.9 / 29.8 / 26.8 GiB**, and our own PLE is **48 GiB**, not
47.7 — so the honest saving is **~18 GiB (INT4) or ~21 GiB (NVFP4)**, which is exactly what our TODO
already estimated for `provsalt/…-PLE-NVFP4`. Our FP8 table is the *same size* as their FP8 table; there is
no free win, only a precision trade. The pagination trap is the same shape as this week's other four: a
query that answers confidently without covering what it claims to.
`ple-access-pattern` already says what it buys and does not: the PLE is a **memory-layout** problem —
2,560 useful bytes per token scattered so a 160-byte row costs a 4 KiB page — so a smaller row still faults
one page and decode barely moves. **It buys resident capacity, not speed.** On a 128 GB box that is still
the constraint worth spending: our KV pool measured 176–247k tokens and varies 119–180k across identical
starts (finding 155), and concurrency is gated by it.

Also new and worth a look: `dicksondickson/…-NVFP4-reshard-mtp-fix` (09-07) — an MTP-specific reshard, in a
week when vllm#55357 and #55951 are both open on MTP/hybrid state handling.
`gitcommit90/Qwen3.8-Flash-Next-NVFP4-DenseFP8-One-Spark` (09-06) is tagged `dgx-spark`/`gb10` and based on
`nvidia/…-NVFP4` — i.e. someone is publishing our exact recipe for our exact box — but the repo holds
**2 files and 0 GiB**: announced, weights never uploaded.

Download ranking says where the field's attention is: `nvidia/…-NVFP4` 26,302, `Baekpica/…-Mixed-Quant-
**SSD-PLE**-GGUF` 17,140, `primitive-ai/…-NVFP4` 14,281. Two of the top three are about getting the PLE
out of RAM.

### EXL3 (2026-09-09) — the best published quality curves, on a stack we cannot use

Four EXL3 repos exist for Flash-Next; the reference is **`turboderp/Qwen3.8-Flash-Next-exl3`** (986
downloads, 41 likes, 09-01) from the ExLlamaV3 author, shipping a bitrate ladder as *branches*:
**2.05 / 3.05 / 4.05 / 5.05 / 6.05 bpw**. Requires ExLlamaV3 ≥ v1.4.5 (current v1.4.8, 09-06).

**Why it is worth knowing about even though we cannot serve it:** the model card publishes **KL-divergence
and perplexity curves against the unquantised model across the whole bitrate ladder**. That is the metric
we use (modal top-1 / forced-logprob divergence, findings 153/158/159) and the one the entire NVFP4
ecosystem does **not** publish — NVIDIA gives task scores, nobody gives divergence. Those curves are a free
yardstick for judging whether NVFP4's measured 12.1 % weight error is good or bad *for its bitrate*.

**Why we cannot use it here.** (1) **vLLM has no EXL3 backend** — the quantization registry in our build
lists awq/gptq/marlin/fp8/modelopt/mxfp4/quark/torchao and friends, no `exl3` or `exllama`. A community
fork does serve it (the `verdictai/glm53-flash-exl3-k4` image in vllm#54458), but that is out of tree.
(2) **ExLlamaV3's Blackwell support is actively broken in places**: open issues #307 (illegal memory access
in `coop_autotune.cu` on consumer Blackwell sm_120), #333 (two-GPU layer split decodes gibberish on
sm_120), #245 (TP=2 illegal access in `pg_gather_kernel`). sm_121/GB10 is a rarer variant again and is not
named in any of them. (3) Adopting it means giving up the vLLM stack entirely — the four determinism
overlays, the MTP path, the PLE offload worker and every measurement tool we have.

Note also `wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-**PLE-FP8**-v1`: even an EXL3 build keeps the PLE
table at FP8 as a separate artifact, which says the body/PLE decomposition is architectural rather than a
property of any one quant format.

### Where the 27B checkpoints live now (2026-09-09)

The 30 archived 27B checkpoints and `laguna-s-2.1-dflash-nvfp4` are **no longer on the GB10**. They live
only at `10.0.0.70:/mnt/bulk/gb10/models` (229 GB on a 2 TB-quota ZFS dataset outside the PBS datastore),
each directory carrying a `SHA256SUMS` manifest and a `SOURCE.json`. Content-verified 30/30 by sha256
before deletion; the local free space went 35 GB → **302 GB**.

Three are identified against their publisher and are therefore **re-fetchable**:

| directory | repository | verification |
| --- | --- | --- |
| `qwen38-27b-fp8` | `Qwen/Qwen3.8-27B-FP8` | 43/43 published hashes, all shards |
| `qwen38-27b-radixark` | `RadixArk/Qwen3.8-27B-NVFP4` | 4/4, 3 shards |
| `qwen38-27b-inferact` | `Inferact/Qwen3.8-27B-NVFP4` | 8/8, 7 shards |

The rest are **locally composed** — bodies and heads spliced here (`ours-fp8head`, `radix-body`,
`uns-bf16head`, `dflash2-*`, …) — and exist nowhere else. Losing that archive loses them.

Kept on the box: `qwen38-27b-nvfp4` + `qwen38-27b-dflash2-syvai-w4a16` (the prod 27B pair named in
`vllm-qwen38.service`), and all ten `qwen38-flash-next-*` variants, which are hardlinked onto one ~111 GiB
base. Our production Flash-Next is `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision `7b71922524…`, fetched
2026-08-26 and spot-verified 3/3 against RadixArk's published shard hashes; its own
`qualification-notes.md` records that only the 48 routed-expert layers are NVFP4 W4A4 and that the PLE
tables are the FP8 ones from `Qwen/Qwen3.8-Flash-Next-FP8`.

## 2026-09-09 — two third-party replies that land on our own findings

**vllm#53670, @Suppressor72 (2026-09-08 21:25) — an independent A/B of the trailing-block drop, and it
only half-replicates ours.** Dual RTX 5090, TP=2, Qwen3.8-27B-FP8 hybrid, 1,648-token alignment unit,
in-checkpoint MTP K=2 (always drafting), 3 reps/arm interleaved, fresh boot per rep:

| | drop ON (default) | drop OFF (#53388) |
| --- | --- | --- |
| MTP acceptance | 83.1 ± 2.9 % | 84.2 ± 0.5 % (+1.1 pp, Welch p=0.59) |
| warm throughput | 149.6 tok/s | 241.9 tok/s (**+62 %**) |
| prefix hit rate | 23.8 % | **64.7 %** (+40.9 pp) |

The **direction agrees** with our `mtp-trailing-block-flag` result (−26 % per warm turn with the drop on)
and the hit-rate mechanism is the same. What did **not** reproduce is our *acceptance* loss: we measured
4–6 pp, they see no detectable difference at n=3. They say plainly it is underpowered to exclude a small
effect. **Treat our acceptance figure as configuration- or workload-dependent until someone reproduces it
on a third layout** — the throughput and hit-rate halves are what two independent layouts now agree on.
They also note this characterises the *blanket* opt-out, not the K=0-conditional variant.

**vllm#51782, @xueyangcs (2026-09-09 05:30) — HPC-Ops TopK's contract, answering our question.**
Verbatim: the contract is **set-exact only** — every selected value ≥ every non-selected value; with
ties, *any* valid top-k set is allowed. They do **not** guarantee set-stability (same inputs → same index
set) or order-determinism, because both need an explicit tie-break or a fixed post-sort, at a cost.

That is the same contract our own kernel has, and it is worth reading next to **det-190**: since the QSA
boundary never ties exactly on our traffic (0 in 6,192 selecting rows), "any valid set under ties" cannot
be what is moving our output. A library being set-exact-only is only a determinism problem where ties
actually occur. Neither reply has been answered — both need a go.

### nvidia/Qwen3.8-27B-NVFP4 fetched and fully verified (2026-09-09 14:22)

19 GiB at `10.0.0.70:/mnt/bulk/hf/nvidia--Qwen3.8-27B-NVFP4`, revision pinned to `dbb8f445b3145f8a`,
aria2 rc=0, **19 of 19 files verified against the publisher** (4 by sha256, 15 by git-blob-sha1), 0 bad.
ZFS 1.76 T free of the 2 T quota. The GB10's own disk was never touched.

The first verification pass reported **"4 verified, 15 without a publisher hash"** and we nearly filed
that as the result. It was a gap in our tooling, not in HF's metadata: **HF publishes two hash kinds** —
`lfs.oid` is a real sha256 for LFS files, and every non-LFS file carries a **git blob sha1** in the
entry's top-level `oid`. `hfget.sh` recorded only the first, so `config.json` and `hf_quant_config.json`
— the files where a silent corruption is hardest to spot and most consequential — were unchecked. Both
are now recorded and checked (`hfverify.py`), and `hfaugment.py` backfills a `SOURCE.json` written
earlier. Skill `model-archive` updated with the rule and the trap.

**Why this checkpoint:** its recipe differs from both our local copy and NVIDIA's Flash-Next release —
FP8 W8A8 on the GDN and attention projections, NVFP4 W4A4 on the MoE **and on `lm_head`**, MTP excluded,
ModelOpt 0.47.0.dev80, and **Local-Hessian calibration on 2,048 samples** rather than MSE. The
experiment is *not* "is it faster". It is whether **Local-Hessian calibration rescues the NVFP4 head**,
which we measured at **2.4 % worse NLL, 8 of 8**, against the FP8 head we ship — a bit-comparable A/B
our logprob-divergence tooling answers directly, and one that would revisit an August decision made on a
single checkpoint's evidence.

**Staging is no longer blocked on disk.** The watchdog's premise (~35 GB free) predates the archive run;
the GB10 now has **302 GB free**, so the ~20 GiB copy costs nothing and needs no deletions. It needs the
box, which `scorediv` holds until ~15:05. Queued for the user's go.

### Partial tensor fetch: one layer instead of the checkpoint (2026-09-09)

Our ingress link is the binding constraint on every "get that checkpoint" plan, and it is **~10 MB/s
(~86 Mbit/s)** — measured today at 10.8 MB/s single-stream, with four parallel streams summing to only
**8.5 MB/s**, so the link is saturated and parallelism does not help.

But most questions are about **one tensor**. HF's CDN answers byte-range requests (`206`,
`accept-ranges: bytes`), and safetensors carries a JSON header with every tensor's byte offsets, so a
single tensor can be pulled without the shard: read the index → range-fetch the shard header →
range-fetch only the matching bytes. `/opt/llm/runners/hfpull_tensor.py` does this and writes a valid
safetensors file plus provenance.

Measured: `lm_head` on `nvidia/Qwen3.8-27B-NVFP4` is **682 MiB** inside a 1,879 MiB shard inside a
19 GiB checkpoint. A 75.8 MiB slice came down in **10.5 s** and was **byte-identical (sha256)** to the
same range in the shard we hold on PBS, and the written file loads with correct dtypes and shapes.

**This changes the H100 plan.** The return leg was the worst term — 126 GiB at 10 MB/s ≈ 3.5 h. If the
deliverable is "does Local-Hessian calibration rescue the NVFP4 head", we need the **head**, not the
checkpoint: quantize on the rented box, push to HF (their egress is fast), pull ~1 GB, decide, and only
then pay for the rest. It also means the BF16 reference — 335.3 GiB, which we cannot host — stays where
it belongs, on the rented box, where the divergence measurement should run anyway.

### ModelOpt quantises layer by layer — the H100 box spec drops a lot (checked 2026-09-09)

Read from the ModelOpt source, not inferred:

- **`layerwise_calibrate(model, forward_loop, calib_func, **calib_kwargs)`**
  (`modelopt/torch/quantization/model_calib.py:2052`) wraps *any* calibration function — including
  `local_hessian_calibrate` — in a layer-by-layer strategy. Its helper `LayerActivationCollector`
  (`utils/layerwise_calib.py`) patches decoder layers with a **skip / run / capture** strategy: earlier
  layers are skipped and fed cached activations, the current layer runs, its output is captured for the
  next. Linear in layers, not quadratic.
- **It is resumable.** `checkpoint_dir` + `_CheckpointState` persist per-layer progress to disk, with
  `save_every`. A preempted spot instance resumes at the layer it reached.
- **Export streams too**: `export_dir` (`export/layerwise_export.py`, `unified_export_hf_streaming.py`)
  writes each layer's shard as it goes rather than materialising the output checkpoint in RAM.
- **`get_qdq_activations_from_prev_layer`** switches on the sequential variant, where layer i+1 sees the
  *quantised* outputs of layer i, so accumulated error is compensated rather than ignored.

Two things about Local-Hessian specifically that change the cost model:

1. **It does not mutate weights** (`_mutates_weights = False`). It is a Hessian-weighted **amax grid
   search** — candidates from `start_multiplier` 0.25 to `stop_multiplier` 4.0 in `step_size` 0.1 —
   minimising `dw · H · dwᵀ`. It is *not* GPTQ weight surgery, so layers are independent given their
   inputs.
2. **The Hessian is per block, not per layer.** `hessian_per_block` is `(cin/block_size, block_size,
   block_size)` with `block_size=16` — about **327 KB** per linear at cin 5120, not a 5120² matrix
   (105 MB). And `build_error_func()` frees the buffer unless `keep_buffer`.

**So peak memory is one decoder layer plus the cached calibration activations**, not 335 GiB. The
activation cache is the real driver: 2,048 samples × seq × hidden × 2 B (≈ 43 GB at seq 2048, hidden
5120), held on the host. That is a **single H100 with ~128 GB RAM**, not 8×H100 with 500 GB — and with
`checkpoint_dir` it can run on **preemptible/spot** pricing. Disk still needs ~470 GiB (335 in + 126 out).

One caveat worth checking before booking anything: `_register_local_hessian_input_hooks` has a
fused-MoE-experts path keyed on `_current_expert_idx`, and weights it cannot pair (conv,
`SequentialQuantizer`, non-eager experts) **fall back to plain MSE with a warning**. On a 512-expert MoE
that fallback is the difference between doing the experiment and thinking we did it — grep the run log
for `local_hessian:` warnings before trusting the output.

### lhbench (2026-09-09): the Local-Hessian weight search is trivial; asking for it the obvious way is a SILENT NO-OP

**The trap first, because it would have cost a multi-day run.** Setting
`cfg = NVFP4_DEFAULT_CFG; cfg["algorithm"] = {"method": "local_hessian"}` — the obvious way to ask for
it — **does nothing at all, and warns about nothing**. `NVFP4_DEFAULT_CFG` carries
`block_sizes {-1: 16, 'type': 'dynamic'}`: block scales are derived at runtime, so a per-block scale
search has no parameters to optimise. ModelOpt prints its three normal progress lines and then
`MSE weight calibration: 0it` — zero modules calibrated — and reports `Calibration complete.` A full run
this way produces a checkpoint identical to plain max-calibration, with nothing in the log to say so.

The correct entry point is the purpose-built config, which exists:
`mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`, carrying
`algorithm={'method':'local_hessian','fp8_scale_sweep':True}` and static block scales. With it the same
matrices give `MSE weight calibration: 32it [487 it/s]` and 0 fallbacks. **Verify by the iteration count,
never by the "Calibration complete" line.**

**What was measured** (GB10, modelopt 0.46.0, torch 2.13.0+cu130, real BF16 layer 24 of
`Qwen/Qwen3.8-Flash-Next` fetched by byte-range, 4.83 GiB, 512 experts of `gate_up_proj [1280,2560]` +
`down_proj [2560,640]`, 32 real expert matrices sampled, ~80 tokens/expert from top_k 10 of 512):

| | |
| --- | --- |
| weight-scale search, 32 expert matrices (0.079 B params) | **0.1 s** |
| extrapolated, 512 experts × 48 layers + dense | **~6 min for the whole model** |
| MSE fallbacks | **0** |

**So the amax search is not the cost, and the question I set out to answer turns out to be the wrong
question.** What this does *not* measure, and what will actually dominate a real run: the **forward
passes** that collect the activations and Hessians (2,048 samples through a 125 B MoE — the harness feeds
synthetic activations straight to isolated `nn.Linear`s and skips the model entirely) and the **I/O** to
stream 335 GiB of BF16 weights. Rough shape, *not measured*: ~9.3 h of unattended fetch at our ~10 MB/s,
plus order 1–3 h of forward passes, plus minutes of search. That still lands in "do it locally", but on
the strength of an estimate, not this measurement.

**One-time cost worth knowing:** the first `local_hessian` call JIT-builds `modelopt_cuda_ext_fp8`
(21.7 s here), then caches it.

**Harness honesty.** This took four iterations, each a real defect: (1) `dim()==2` filtering skipped the
fused 3-D expert tensors and timed 3 % of the layer, the wrong 3 %; (2) rank-3 detection then swept in
`linear_attn.conv1d.weight [10240,1,4]`, setting `n_experts=10240` and emitting 16 "input features (4) not
divisible by block_size" fallbacks that were conv slices, not experts — experts must be found **by name**
(`.experts.`); (3) every matrix was fed the full token budget when a real expert sees ~2 % of tokens
(top_k 10 of 512), overestimating the part that dominates by ~51×; (4) the silent no-op above. Only (4)
would have survived into a production run undetected.

### headcmp (2026-09-09): Local-Hessian calibration DOES change the scales, and reconstructs better

Activation-free test of the necessary condition: did NVIDIA's Local-Hessian calibration actually choose
different NVFP4 scales on `lm_head` than plain max would? Byte-range fetch of the BF16 head from
`Qwen/Qwen3.8-27B` (2.4 GiB, 3 min) against `nvidia/Qwen3.8-27B-NVFP4`'s published head — no checkpoint
downloaded on either side.

**Format validated first, before reading anything into the numbers.** Dequantising the publisher's own
packed weights with the publisher's own scales reproduces the BF16 head to **8.49 %** relative error
(E2M1 levels {0,.5,1,1.5,2,3,4,6}, **low nibble first**, group size 16 = 5120/320, global
`weight_scale_2` 1.2716e-4). The high-nibble-first ordering gives 141 %, which is how we know the
ordering rather than assuming it. Only with that confirmed is a scale comparison meaningful.

| scale set, same BF16 weights re-quantised to NVFP4 g16 | relative reconstruction error |
| --- | --- |
| publisher's shipped packed weights + LH scales | 8.482 % |
| re-quantised with **Local-Hessian** scales | **8.482 %** |
| re-quantised with **plain-max** scales (`amax/6`, FP8) | **9.483 %** |

**Local-Hessian is ~1.0 pp better, ~10.6 % relative**, and it changes **69.77 %** of group scales.
Direction: LH picks a **larger** scale in 73.3 % of the differing groups (median LH/max ratio **1.385**,
p10 0.909, p90 1.600) — i.e. it does not simply clip tighter.

**What this does and does not establish.** It establishes the necessary condition: LH is not a no-op on
this matrix, so the calibration axis is real and worth pursuing. It does **not** establish output
quality — this is plain L2 weight error, which is not even what LH optimises (it minimises a
Hessian-weighted error). That LH also wins on plain L2 is a bonus rather than the design target, and the
honest next step is output error under **real activations**, which needs final hidden states captured
from a forward pass — tooling we do not have yet.

**Caveat on the baseline:** "plain max" here is reconstructed as `amax/6` rounded through FP8 with the
global scale_2. That reproduces the publisher's dequant path exactly and agrees exactly on 30.19 % of
groups, so it is the right formula in kind — but if it were subtly suboptimal, LH's margin would be
flattered. The 8.482 % figure is anchored to the publisher's own bytes and does not depend on it.

Runner `headcmp.sh`, tools `lh/headcmp.py` (scale census), `lh/headval.py` (format validation),
`lh/headq.py` (like-for-like re-quantisation). `headcmp.py` crashed on `quantile()` over 55 M elements —
subsample before quantiles; `headq.py` does.

### 2026-09-09: #53142 / #54173 / #52244 are ONE defect, and upstream already closed the root cause

@Windless84 root-caused it on 09-09 (on our exact model, RTX PRO 6000) and posted the same finding to
all three threads. `EngineCore._initialize_kv_caches` set
`cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups)`. The QSA indexer
registers a `CircularBufferSpec` whose block size is the **ring capacity — 4 tokens without spec decode,
8 with MTP-3** — so the global `cache_config.block_size` came out 4/8 while the Mamba managers run on
**1,600**. Two consumers trust it:

1. `mamba_hybrid.add_request` seeds the resumed state column with
   `(num_computed_tokens - 1) // cache_config.block_size` → the index lands past the Mamba block table
   on any resume over a cached prefix. That is the illegal memory access of **#54173**, and why it fires
   on the *second* request of a shared-prefix pair and never on a fresh prompt.
2. `Scheduler._mamba_block_aligned_split` aligns chunk ends to it → every chunk reaches the align-mode
   managers unaligned. Their measurement: 140K tokens took **186 of 240 blocks instead of 113**.

This is exactly our post-mortem row A ("align-mode block size = QSA ring capacity instead of the mamba
block"), arrived at independently and with the mechanism named one level deeper than we had it.

**Upstream closed it on 2026-09-04 with #53906**, which excludes non-prefix-cacheable groups from the
`min(...)` — verbatim: *"their small block_size would otherwise drag the global block_size below the real
allocator block size and desync it from mamba."*

**Checked against our own stack, and it is already in.** `vllm-venv-fnmain2`:

| | |
| --- | --- |
| `v1/engine/core.py:338` | carries the #53906 comment; the `min()` at 345 filters on `g.kv_cache_spec.prefix_cacheable` |
| `mamba_state_block_size` (the #54076 identifier) | **0 occurrences** — #54076 is NOT applied |
| `_mamba_block_aligned_split` | still reads `self.cache_config.block_size` … |
| … which now resolves to | **1600** — `Setting attention block size to 1600 tokens`, confirmed in three separate fnmain2 server logs (09-08 21:24, 09-09 14:37, 14:52) |

So **the align defect is closed on our current build by a different fix than the two we were tracking**.
#54076/#53798 address the same defect more robustly (reading the mamba group's own spec instead of
trusting the global min), but they are not load-bearing here any more. Our determinism narrative should
stop listing "align block units (#54076/#53798)" as a live defect on fnmain2 — it was live on the
**preview** stack where we measured it, which is what our 09-09 withdrawal comment said in the past
tense, so no upstream correction is owed.

### arXiv 2608.28113 — "H-Scale: Hessian-Guided Scale Refinement for NVFP4", Qwen Team (read 2026-09-09)

The paper for the thing we spent today measuring, and it is by **the model authors** (Hao Yu, Zheng Li,
Dayiheng Liu, Jianwei Zhang; Qwen Team, Alibaba).

**Their claim, in our terms:** per-group scale selection is *the* bottleneck for NVFP4 fidelity, and
existing PTQ refines quantized weight *values* while leaving the scales to RTN. H-Scale post-processes
the scales, choosing hardware-valid FP8 group scales that minimise a **Hessian-weighted** reconstruction
error from calibration activations — "targeting layer output perturbation more directly" — by enumerating
**K = 16 neighbouring FP8 candidates** around the baseline and keeping the best. Drop-in, **zero
inference overhead**.

**Why this matters to our roadmap, point by point:**

- It is the same *idea* as ModelOpt's `local_hessian` (diagonal second-order proxy, candidate
  enumeration over FP8 group scales), though **not necessarily the same implementation**: ModelOpt
  sweeps `start_multiplier` 0.25 → `stop_multiplier` 4.0 at `step_size` 0.1 ≈ 38 candidates, wider than
  their K=16. Do not treat their numbers as ours.
- **They evaluate on an MoE and it works** — Qwen3-30A3-Thinking and -Instruct (30 B total, 3 B active).
  That is the direct counter to the worry raised by NVIDIA choosing MSE for Flash-Next: the technique is
  not known-bad on MoE. It does *not* answer whether **ModelOpt's** fused-expert code path engages
  correctly, which stays gate 1.
- **The headline is the shape we want:** GPTQ+H-Scale reaches **81.22 on Qwen3-30A3-Thinking against a
  BF16 baseline of 81.06** — i.e. the quantised model at or above BF16 on their average. And a scale-shift
  experiment on MLP projections removes ~**50.6 %** of the initialisation error.
- **It composes** with weight-rounding methods (GPTQ, ArcQuant, MR-GPTQ, GPTAQ, 4over6) rather than
  replacing them — so "better calibration" and "better rounding" are orthogonal axes.
- **Gate 0 is measuring exactly what they optimise.** Our `headcmp` result — LH changes 69.77 % of group
  scales and is 1.0 pp better in *plain weight* space — is the weaker metric; the paper's objective is
  layer **output** perturbation, which is what `headcap` captures activations for. The paper predicts
  gate 0 should pass, and predicts the output-space margin should be *larger* than the weight-space one.

**Timeline note that partly resolves the NVIDIA puzzle.** Their Flash-Next build used ModelOpt from
2026-08-24 with MSE, four days *after* the 08-20 toolchain they used with Local-Hessian on the 27B. An
August 2026 paper is recent enough that adoption lag is at least as likely an explanation as a measured
negative result on MoE — but that is a guess, and gate 1 is still the thing that decides it for our
pipeline. Hardware there is H200/Blackwell, not GB10; NVFP4 is the same format.

### The real reason NVIDIA used MSE for Flash-Next: expert COVERAGE, not missing support (2026-09-09)

The user's correction, verified against the installed ModelOpt 0.47 source, and it resolves the puzzle
completely. My earlier guess — "Local-Hessian may not support the fused-MoE path" — is **wrong** and is
withdrawn. Two facts settle it:

**1. NVIDIA's Local-Hessian build has no experts at all.** `nvidia/Qwen3.8-27B-NVFP4`'s
`quantized_layers` are `gate_proj 64 / up_proj 64 / down_proj 64` — one each per layer, i.e. a **dense
MLP** — plus attention and `lm_head`. Zero expert modules. They applied Local-Hessian to a model with no
routing at all.

**2. The expert-coverage knob does not exist for fused experts.** `moe_calib_experts_ratio`
("force forward tokens to % of experts during the calibration pass") is defined on
**`_QuantSparseSequentialMoe` only** — checked class by class:

| class | has `_moe_calib_experts_ratio` |
| --- | --- |
| `_QuantSparseSequentialMoe` | **yes** |
| `_QuantFusedExperts` / `_QuantNonGatedFusedExperts` | no |
| `_QuantQwen3VLMoeTextExperts`, `_QuantLlama4TextExperts`, `_QuantDbrxExperts`, `_QuantGptOssExperts` | no |

Its own docstring says so: *"Not supported for all MoE architectures; currently works with a few
HuggingFace models such as Mixtral, Qwen3Moe, MiniMax"*, and the sequential base class notes
*"Transformers>=5.0 has batched experts, no per-expert quantizers."* **Flash-Next is batched/fused**
(`gate_up_proj [512,1280,2560]`), so the knob is unavailable to us.

**So the honest statement of the risk.** MSE optimises expert scales without needing to know which
tokens reach each expert. Local-Hessian needs `H = XᵀX` **per expert**, from representative routed
tokens. On a fused-MoE model ModelOpt gives no way to force routing coverage, so each expert's Hessian
is estimated from whatever the calibration corpus happens to route to it. That is a far better
explanation of NVIDIA's choice than anything about support, and it is architecture-specific in exactly
the way their two builds differ.

**The arithmetic is not obviously against us.** 512 experts, top_k 10: 4.19 M calibration tokens
(2,048 × 2,048) give **~81,900 tokens per expert on average**. The mean is comfortable — but routing is
skewed and the mean says nothing about the tail, and the tail is precisely what Local-Hessian needs.

**Gate 1 is therefore reframed.** Not "does the fused path engage" — the user cites the 0.46 changelog
(2026-08-17) saying the fast Local-Hessian path is used automatically for dense *and* fused-MoE expert
weights with a ~34× Triton sweep, and the Triton fast path is present in the 0.47 source we installed.
The new gate 1 is: **measure the per-expert token-count distribution our calibration corpus actually
produces on Flash-Next**, using the same probe technique as `headcap` on the routing gate. If the
minimum-coverage expert still sees thousands of tokens, the concern dissolves; if some see ~0, we learn
which and fix the corpus — and that is a data problem we can solve, not a toolchain limitation.

### GATE 0 PASSES: Local-Hessian's advantage is ~3× LARGER in output space than in weight space

`headcap` captured **11,905 real `lm_head` input vectors** (5,120-dim) from the served 27B — the probe
inserted at `qwen3_5.py:408`, reached by delegation from `qwen3_vl.py:2969`, with `prompt_logprobs` set
so `compute_logits` sees every prompt position rather than only the last. Venv restored byte-identical.

Relative **output** error `‖X(W−Ŵ)ᵀ‖_F / ‖X Wᵀ‖_F` on those activations, same BF16 head re-quantised to
NVFP4 g16 under each scale set:

| scale set | output error | weight error (headcmp) |
| --- | --- | --- |
| **Local-Hessian** (NVIDIA's published scales) | **3.067 %** | 8.482 % |
| plain-max (`amax/6` through FP8) | 4.370 % | 9.483 % |
| **LH better by** | **1.303 pp — 29.8 % relative** | 1.001 pp — 10.6 % relative |

**The margin grows in output space, which is the whole claim of the method.** Local-Hessian minimises a
Hessian-weighted error, so plain L2 weight error understates it; measured against real activations the
advantage is roughly **three times larger in relative terms**. Note also that output error (3–4 %) is
far below weight error (8–9 %) — the activations simply do not excite every weight direction equally,
which is precisely the premise Hessian weighting exploits.

This sits at the top of the user's predicted range for output reconstruction ("clearly better, perhaps
20–30 % lower perturbation"), and it is consistent with the Qwen team's own H-Scale result
(arXiv 2608.28113).

**Computed without materialising the residual**: `‖X Dᵀ‖²_F = Σⱼ dⱼᵀ (XᵀX) dⱼ`, so one 5120² Gram matrix
turns an 11,905 × 248,320 product (11.8 GB) into a batched quadratic form over weight rows.

**Limit, stated plainly:** X comes from our **quantised** 27B body and **our** prompts, not a BF16 body
and not NVIDIA's Nemotron calibration set. This answers "which scale set is better at deployment time on
our traffic", not "did we replicate their recipe".

### GATE 1 PASSES: expert routing coverage is not the problem — but the headline number is 48× inflated

`expcov` (2026-09-09 22:04–22:17). An env-gated bincount of `topk_ids` in the **concrete**
`apply()` of `flashinfer_cutlass_moe.py:247` — deliberately not `FusedMoEMethodBase.apply()`, which is
abstract and would have produced a clean-looking zero. Venv restored byte-identical. 33 prompts
(30 real agent prompts + prose), **160,196 prompt tokens**, MTP off, 12,000 `apply()` calls,
**74,936,640 routed (token, expert) pairs**.

| summed over all 48 layers | |
| --- | --- |
| mean per expert | 146,361 |
| **min** | **78,481** |
| p1 / p10 / median / p90 / max | 83,942 / 105,752 / 139,998 / 191,491 / 287,763 |
| experts with ≤ 10,000 assignments | **0 of 512** |
| skew | max/min **3.7×**, p90/p10 **1.8×** |

**The correction that matters.** Those counts are summed across layers, and Local-Hessian estimates
`H = XᵀX` **per (layer, expert)** — layer 3's expert 7 is a different matrix from layer 24's expert 7.
So the number the method actually uses is **48× smaller**:

| | mean | worst expert |
| --- | --- | --- |
| per expert, summed over layers *(the flattering number)* | 146,361 | 78,481 |
| **per (layer, expert)** *(the real one)* | **3,049** | **~1,635** |
| scaled to a real 4.19 M-token calibration (26.2× our corpus) | ~79,835 | **~42,809** |

**Verdict: PASS, and comfortably.** Even on the conservative per-layer reading our small 33-prompt corpus
already gives the worst-covered expert ~1,600 samples, and a real calibration run gives it ~43,000. The
routing is also remarkably *even* — 3.7× between best and worst expert, 1.8× between p10 and p90 — so
there is no starved tail to rescue. **Zero experts fall below 10,000 even before the per-layer division.**

**One assumption left unmeasured**, stated rather than buried: the per-layer *minimum* above assumes the
routing skew is similar in every layer, because the probe summed layers rather than keying by layer. If
one layer routes far more unevenly than the rest, its worst expert could sit below the ~1,635 estimate.
Keying the bincount by layer index is a small change to the same probe and is the natural refinement if
gate 2 gives an ambiguous answer.

**So the coverage hypothesis is answered for our corpus, and it is not what stopped NVIDIA.** The
architectural facts stand — `moe_calib_experts_ratio` is unavailable for `_QuantFusedExperts`, and their
Local-Hessian build was dense — but on Flash-Next with our own agent traffic, natural routing supplies
ample per-expert statistics without any coverage forcing.

### GATE 2 PASSES: Local-Hessian is ~22 % better on real EXPERT matrices — CORRECTED

**The first version of this entry was contaminated and its numbers are withdrawn.** Original claim:
"median 24.3 %, range 20.3–53.5 % across 15 experts". Corrected below.

`expact2` captured **16,384 BF16 expert inputs (2,560-dim)** from `layers.24.mlp.experts` — hooked at
`Qwen3NextSparseMoeBlock.forward` before any quantisation, server in `--enforce-eager` because that
forward is fullgraph-compiled. Routing recomputed offline from the same layer's own
`mlp.gate.weight [512,2560]`. Venv restored byte-identical.

**The contamination: half the captured rows are batch padding, and padding routes to experts 0–9.**
8,192 of 16,384 rows are all-zero. A zero row softmaxes to a *uniform* distribution, so `topk` returns
the ten lowest indices — every padding row lands on experts 0..9. Those then look like the
best-covered experts (8,192–9,040 rows each) while their **real** counts are 242, 6, **0**, 128, 145,
198, 848, 106, 50, 21. Selecting "the 16 best-covered experts" selected exactly the artifact: expert 2
had *zero* real rows (hence its NaN — `‖X Wᵀ‖ = 0`), and expert 1's headline 53.5 % gain rested on **six**
rows.

**Corrected method:** drop all-zero rows, then take every expert with ≥ 300 genuinely routed rows.

| | relative output error ‖X_j(W−Ŵ)ᵀ‖ / ‖X_j Wᵀ‖ |
| --- | --- |
| plain-max | 6.99 – 7.83 % |
| **Local-Hessian** | **5.42 – 6.08 %** |
| **LH better by, 32 experts (517–2,396 real rows each)** | **min 20.2 %, median 22.2 %, max 24.1 %** |

**Tighter than the contaminated version, and more credible for it** — 32 independent experts landing
within four percentage points of each other, against the head's 29.8 % at gate 0. The calibration
advantage is real on the matrices carrying the 73.3 GiB; it is somewhat smaller on experts than on the
head, which is itself worth knowing.

ModelOpt chose the LH scales (`NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`, 0.47.0rc1.dev31, calibrated on each
expert's own routed rows — calibration iterations observed, so not the silent no-op); plain-max is
`amax/6`; **both scale sets go through one quantise/dequant path**, isolating the scale choice. Output
error via the Gram identity, never materialising the residual.

**Remaining limits:** our agent corpus through the quantised body, one layer of 48, activations from the
eager path (same mathematics, different kernel fusion). With padding removed the capture yields 8,192
usable rows and a median of 88 real rows per expert, so the 32 measured experts are the better-covered
tail of one layer.

