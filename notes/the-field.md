> **Superseded, 2026-08-27.** This page opened by saying no other project was a single-box vLLM
> deployment, "which is the only reason this repo exists". That stopped being true within a day.
> `starkweatherdigital` and `getrefined` both run vLLM on a Spark, `0xBakeer` added a vLLM recipe
> alongside their llama.cpp one, and `SirTificate` runs vLLM on 2x RTX PRO 6000. The field went
> from ~9 projects to ~35 in 48 hours. Treat every count and ranking below as a snapshot.
>
> What still distinguishes this repo is narrower and worth stating honestly: a kernel-level
> profile of where single-stream time actually goes on GB10, concurrency measured past c=8, and
> the failure-mode catalogue. Not the deployment itself.


| repo | stack | hardware | result |
|---|---|---|---|
| [MiaAI-Lab/…-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | SGLang TP2, NVFP4 | **2×** Spark, ConnectX-7 200 Gb RoCEv2 | **64 tok/s** single-stream, 117 aggregate @ c=2 (NEXTN 3/1/4) |
| [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp, mmap tensor pinning | 1× Spark | ~22 tok/s (measured `--parallel 1`) |
| [lendome/llama.cpp-qwen4exp](https://github.com/lendome/llama.cpp-qwen4exp) | llama.cpp + PR #27742 | — | build recipe |
| [hocestnonsatis/qwen3.8-flash-next-16gb](https://github.com/hocestnonsatis/qwen3.8-flash-next-16gb) | llama.cpp UD-IQ1_S | 16 GB VRAM | extreme-quant route |

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

**This is the most important open question for this repo.** The justification for a vLLM route
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
  For us this only re-scopes a sentence: **#53896 is the only place the *vLLM* implementation
  exists**, which remains true; llama.cpp is a separate implementation and is now merged.
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
| `flashinfer_b12x` | rejected — `not supported for unquantized MoE`; the MTP drafter's MoE is unquantized and `--moe-backend` is global. With MTP off it faults with an IMA |
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
