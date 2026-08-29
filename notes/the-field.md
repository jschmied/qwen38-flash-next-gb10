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
| [0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark) | llama.cpp, mmap tensor pinning | 1× Spark | ~22 tok/s, `--parallel 1` |
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
