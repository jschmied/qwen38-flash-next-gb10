## What could be better

Three things we measured on another GB10 with the same image (`vllm/vllm-openai:qwen38-flash-next`) that look directly applicable to this kit. Numbers are from https://github.com/jschmied/qwen38-flash-next-gb10 (finding numbers refer to `notes/prefill-investigation.md` there); prompts are 7,503 and 29,263 tokens, prefix caching off, two or three server starts per cell.

**1. `MAX_NUM_SEQS=4` is the aggregate-throughput ceiling, not the PLE offload.** With the offloaded PLE we saw aggregate decode keep scaling well past 4 streams: 266.8 tok/s at 48 concurrent streams on the baseline checkpoint (short prompts, MTP off, one run per level — a shape, not a calibrated curve), and the swap/major-fault cost per token *fell* about 4× under load because neighbouring lookups share pages. Your 85.9 tok/s at 4 streams with a 736k-token KV pool leaves most of that on the table; 16 or 32 is worth one run.

**2. Prefill: same image, ~1.8× lower TTFT with a different configuration.** Your table: 5.00 s at 8k (1,646 tok/s), 15.83 s at 32k. Ours on this image: 2.7–2.8 s at 7,503 tokens and 10.6–10.9 s at 29,263 (finding 69/74). The differences, in the order I would A/B them because the first two cost nothing:
- `compilation-config mode 0` here vs torch.compile on (we keep cudagraphs at `FULL_DECODE_ONLY` as you do);
- `MAX_NUM_BATCHED_TOKENS=2048` vs 4096; 8192 was another −7 % at 32k for us but loses on ~4k agent turns, so 4096 is the safe default (finding "prefill batch is workload-dependent");
- dense projections: our checkpoint runs FP8 blockwise on CUTLASS with no fallback, yours runs MXFP8 through FlashInfer with a BF16 emulation fallback for every `N % 32 != 0` layer — those emulated layers are probably a large part of the gap, and only a per-layer profile on your side can say how much;
- PLE tables in anonymous memory via the vllm#53899 offload worker vs `MADV_RANDOM` mmap of the packed table (your page-cache design is the nicer memory shape; whether the random faults show up in prefill is one `perf`/`majflt` check away).

One cheap test on top: on this image with a blockwise-FP8 checkpoint, stock TTFT at 8k is bimodal (5.03 s vs 3.51 s) depending on whether the scheduled chunk's token count is a multiple of 4 (finding 69/95, a CUTLASS sm120 dispatch issue — vllm#55180 has the analysis). Your 5.00 s is the shape of the slow mode. If any of your dense layers hits that kernel, padding the prompt by 0–3 tokens changes the number; if none does, nothing happens.

**3. Warm turns (agent use).** Two independent fixes:
- with MTP and prefix caching, vLLM drops the trailing prefix block on every turn; `disable_eagle_block_drop` (vllm#53388, main only) took our warm agent turn from 2.05 to 1.52 s (−26 %, three starts, acceptance unchanged). On this image it is a 4-file port; ours is in `tools/main/`.
- the hybrid GDN/QSA cache aligns hits to 1,600-token blocks, so a re-prefill of the tail past the last aligned block is paid on every warm hit. Padding the shared system prefix to a multiple of 1,600 tokens took a +130-token warm hit from 0.75 s to 0.27 s in our probe (findings 97/98), with no server change.

## Proposed change

- `.env.sample`: `MAX_NUM_SEQS` higher than 4 once you have run one concurrency sweep on your KV budget; a comment naming the ceiling you measured.
- `start.sh`: make the compile mode a knob (`COMPILE_MODE`, default whichever wins your A/B) and `MAX_NUM_BATCHED_TOKENS` default 4096.
- README: a "warm turns" note with the align-block padding trick, and the #53388 port if you want it — happy to send that as a PR against your patch generators if useful.

Everything above is measured on one GB10; treat it as a starting point for your own A/B, not as a promise. Thanks for the MXFP8 dimension table and the `MADV_RANDOM` packed-table design — both go into our notes.
