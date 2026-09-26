POSTED 2026-09-26 as https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/78 — the user's go ("edit it: answer stays, other stuff goes to new issue", 2026-09-26). New issue on MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (Improvement template).

Title: v0.30 lane: three measured levers from our GB10 stack (cold-page readahead for the mapped PLE, NVFP4 draft head, GDN RecoverSSM)

## What could be better

Three results from our single-GB10 stack (vLLM nightly `1ea7c63f4` + our overlay, Qwen3.8-Flash-Next, MTP 3) that
map onto the new v0.30 lane:

1. **The file-backed PLE table starts cold.** Loading the weights evicts it from the page cache, and a decode step's
   rows are known only after the previous step samples, so a host prefetch has no lead. The GPU then faults the step's
   ~30–57 missing pages itself, one at a time, until the table warms.
2. **The draft head is read in full precision per draft step**, even with a reduced draft vocab.
3. **The native GDN spec-decode path snapshots the full recurrent state** after every verify token, and reserves a
   Mamba block per draft position. That costs KV and GPU-cache write-back.

## Proposed change

1. **Readahead for the step's pages:** `MADV_WILLNEED` for every page, then `MADV_POPULATE_READ` per page, both via
   ctypes, which releases the GIL. The GPU's faults then land on reads already in flight in parallel. Pure Python:
   [vllm#58835](https://github.com/vllm-project/vllm/pull/58835), on top of our checkpoint-mapped backend
   [vllm#58439](https://github.com/vllm-project/vllm/pull/58439). Your ATS read should have the same cold window.
2. **Quantize the sliced draft-head rows to NVFP4** after the vocab slice (our slice is 32k).
3. **GDN RecoverSSM:** verify from one checkpoint with a small per-token replay record, then commit the accepted tokens
   once after sampling. It needs the V2 model runner, which you already pin. Numbers and code:
   [bilikaz/qwen38-flash-next-recipe#1](https://github.com/bilikaz/qwen38-flash-next-recipe/issues/1) and
   [vllm#56466](https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579).

## Measured impact

All on one GB10, 2 server starts per arm, output hashes compared:

| change | before → after | notes |
|---|---|---|
| 1. cold-page readahead | cold pass 59.15 / 59.41 → **56.80 / 56.50** ms/step; warm 54.73 / 54.71 → 54.95 / 54.75 | 12/12 paired requests faster; major faults per step 29–33 → 0.1; outputs identical. A gated wait + C helper, tried first, was slower ([§4x–4z](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/speed-of-light.md)) |
| 2. NVFP4 draft-head slice | 606 → 45 MiB per draft step; c=1 −3.4 % ms/tok, c=4 +2.8 % tok/s | acceptance slightly up; c=1 hashes identical ([finding 234](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/prefill-investigation.md#L3886)) |
| 3. GDN RecoverSSM (`align` + prefix caching) | agent loop 1.31 → **1.10** s/turn; KV tokens in the same memory **+37 %**; per verify cycle −2.4…−5.2 % | text differs from the native path by summation-order size (first greedy divergence at a median of 26 tokens, against 31 for a reduction-order change) ([§4t–4u](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/speed-of-light.md#L661-L729)) |

One measurement caution from the same runs: c=4 greedy text on this model is not stable run to run. Two of our ~10
c=4 passes drifted to different text at ~10 % different throughput, so we compare concurrency cells only where the
output hashes match.

*Measured with AI assistance (Claude); numbers are from the linked data.*
