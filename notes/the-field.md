# Who else is running this, and how

As of 2026-08-26, the day of release. None of these is a single-box vLLM deployment,
which is the only reason this repo exists.

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
