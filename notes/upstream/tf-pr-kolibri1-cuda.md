POSTED 2026-10-03 as #328. Sits on #327, whose commit is the first one here. CONTRIBUTING asks for an issue before a new family. Kolibri 1 came out today, and this PR is meant as that conversation, with the code and receipt attached.

## What this changes

[Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) (`kolibri1`) is a 78B MoE with 3.5B active parameters. A new CUDA family reads its FP8 checkpoint as it ships: e4m3 weights in 128 x 128 blocks with fp32 scales, and bf16 embeddings, head, norms and router. It runs on one GPU, serially or with `--parallel N`.

- **Architecture**, after Aleph Alpha's vLLM plugin (`aleph_alpha_inference/kolibri1.py`):
  - GQA with 48 query heads, 4 KV heads and per-head q/k RMSNorm;
  - 40 sliding layers with RoPE and 513 keys, the row's own included, and 10 full layers with no positions;
  - extra norms after attention and after the MoE;
  - every layer MoE: top 6 of 384 on logit + expert bias, weighted by the unbiased `sigmoid(logit)` with no renormalisation, plus an ungated shared expert.
- **`tensorfold/cuda/fp8/experts`**: grouped block-FP8 experts on `cuda.experts`' plan, the shared expert last in the table.
  - A warp covers 32 columns, which lie inside one 128-row scale block.
  - Each 128 inputs run a bf16 mma chain from zero. The chain then folds into fp32 as `fma(chain, scale, acc)`, in block order.
  - e4m3 converts to bf16 exactly, subnormals included.
  - Packing only reorders the checkpoint's bytes into the lanes' order.
- **Sliding attention**: one Triton program per (row, key head). It walks the row's own 513 keys in 64-key tiles by absolute position, from a ring of `PROMPT_CHUNK + 513` keys per stream.
  - Full layers keep every position and use the shared `prefill_attention` (prompt) and tree `attention` (decode).
  - The 40 sliding layers take 215 MB a stream whatever the context. That puts 4 streams of 262,144 tokens in 20.8 GiB.
- **`--parallel N`**: the shared `Scheduler` drives a decoder that runs a prompt chunk a round (1,024 rows while others decode), then one row of every decoding stream in one forward.
  - Each stream has its own cache slot.
  - The context is sized from `capacity.available_bytes`, up to 262,144. An explicit `--context` that doesn't fit is refused with the size that fits.
- **Exact rows**: every kernel is row-invariant. That covers the FP8 lane matmul for projections, the grouped experts, the `moe.router` kernel for the router and the LM head (fp32 logits), and a per-row RMSNorm. So a stream equals its solo run.
  - Prompt and decode rows differ only in the full layers' attention kernel, as on the other engines.
- **CLI**: the startup line reads `engine.drafts` when the engine sets it, so Kolibri says "drafts: off". Kolibri ships no draft head.
- **Refused for now**: `response_format` grammars, `--tp 2` and `--drafter`, each with a message.
- **Not here yet**: a prefix cache, CUDA graphs and a draft path. Kolibri decodes one row a round, so the lane contract's drafted == serial holds trivially.

## Receipt

Environment: TensorFold `main` at 609ca41 plus these commits; PyTorch 2.13.0+cu130 and Triton 3.7.1; one DGX Spark (GB10, sm_121, driver 580.178.04). Checkpoint `Aleph-Alpha/Kolibri-1` @ e52eb462, checked against the publisher's sha256 for all 40 files. Served with `tensorfold serve … --parallel 4`, with the context chosen automatically (4 x 262,144).

**Exactness**, `tools/bench_concurrent.py … --alone --serial`:

| prompt | temperature | aggregate tok/s (4 streams) | steady tok/s | alone | serial |
|---|---|---:|---:|---|---|
| code | 1.0 | 99.7 | 102.1 | 12 equal / 0 unequal | 4 / 0 |
| chat | 1.0 | 100.7 | 105.2 | 12 / 0 | 4 / 0 |
| code | 0.0 | 112.1 | 115.2 | 12 / 0 | 1 / 0 |
| chat | 0.0 | 109.5 | 114.7 | 12 / 0 | 1 / 0 |

**Decode**, `tools/bench_openai.py`, one stream, median of 5:

| prompt | temperature 1.0 | temperature 0.0 | TTFT |
|---|---:|---:|---:|
| fibonacci-raw | 40.3 tok/s | 40.6 tok/s | 0.06 s |
| gpu-chat-no-think | 40.3 tok/s | 40.6 tok/s | 0.12 s |

**Cold prompts**, `tools/prefill_cold.py`, median of 3:

| 2k | 8k | 16k | 32k | 64k |
|---:|---:|---:|---:|---:|
| 2,410 tok/s | 2,329 | 2,235 | 2,087 | 1,856 |

There is no earlier TensorFold build to compare against. For scale, a field report from one DGX Spark today used vLLM main plus Aleph Alpha's plugin with a bf16 cache ([pulseandthread/kolibri-1-dgx-spark](https://github.com/pulseandthread/kolibri-1-dgx-spark)), on its own client and prompts, so the numbers aren't like for like:
- one stream: 50.1 tok/s;
- 4 streams: 111.6 tok/s aggregate;
- prompts: 3,609 tok/s at 4.9k and 4,615 at 19.8k.

So this engine matches it at 4 streams, and trails on one stream and on prompts. The prompt projections run on the exact W8A16 lane matmul, and nothing is captured in CUDA graphs yet. Those are the next changes.

**Quality.** `tests/cuda/kolibri1_reference.py` is an fp32 forward from the FP8 weights. On a German chat prompt (reasoning off) both reply "Die Hauptstadt von Frankreich ist Paris." token for token; KL(reference ‖ engine) is 0.0014 on the first row, below 0.0001 after. Over HTTP, a needle in a 90,965-token prompt came back exact. Thinking replies, `reasoning_effort: none` and a tool call (Hermes `<tool_call>`, parsed into `tool_calls`) all worked.

**Tests:**
- `tests/cuda/test_fp8_experts.py` (3), `test_kolibri1_attention.py` (5) and `test_kolibri1_forward.py` (4): 12 passed on the GB10. They cover the dense reference, a pair's bits across calls, the masked softmax at windows 513, 64 and 1, a wrapped ring equal to full caches bit for bit, the fp32 reference on a tiny checkpoint, chunked prompts equal to whole ones bit for bit, three streams equal to each alone bit for bit, and a prompt longer than the ring.
- `tests/test_family*.py`, `tests/test_cli*.py` and `tests/test_quant_family_formats.py`: 191 passed, 6 skipped. One failure, `test_family_prefill_step`, needs `mlx-lm` and fails the same way on `main`.

**Not run:**
- other GPUs (RTX PRO 6000, sm_89–10.x); the kernels use bf16 mma and the shared `MIN_CAPABILITY`;
- a Mac (no MLX family here);
- two ranks;
- a quality sweep beyond the reference checks above.
