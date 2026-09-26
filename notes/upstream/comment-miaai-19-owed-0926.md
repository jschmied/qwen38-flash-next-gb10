POSTED 2026-09-26 as https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/19#issuecomment-5846690586 — the user's go ("post to mia and bilikaz", 2026-09-26). MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark #19 comment.

Two things we owed this thread since 09-09, and three results from our box that may be useful for the v0.30 lane.

**1. The vllm#55533 check we promised: it does not reproduce here** ([finding 155](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/prefill-investigation.md#L1692)).
- We polled `vllm:num_requests_running` at c=8 with MTP 3: median 5–6, and it reached **8** in every arm.
- A no-spec control ran at the same time. MTP stayed faster at c=8 (23.4–25.3 vs 19.9 tok/s), so there was no
  collapse to ~3 sequences.
- The `1 + k` Mamba-block charge behind their formula is real, visible in the attention block (1,568 / 1,600 / 1,616
  tokens at k = 0 / 3 / 4). Our pool is simply far from the ceiling that produces their window.
- That agrees with @Nipale-ai's sweep above: no flatline at 4.

**2. The capture-width re-run: small on GB10** ([§5d](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/speed-of-light.md#L940-L959)).
- We had promised c = 3/5/6. We ran the stronger version instead: our own list `[1,2,4,8]` leaves every c ≥ 3 decode
  without graphs at MTP 3, so we A/B'd it against a full list up to 64 tokens. Setup: 2 starts per arm, c = 1/4/8/16,
  outputs hashed.
- With identical text at c=4 the full list is **+2.1 %**; at c = 4/8/16 with distinct prompts the arms overlap.
- The graph-less forward costs little once the step is GPU-bound, and async scheduling hides the launch overhead.
  So your kit's full capture list is right, and cheap (+0.28 GiB of graphs here), but not a big lever.
- One caution from the same runs: c=4 greedy text on this model is not stable run to run. Two of our ~10 c=4 passes
  drifted to different text at ~10 % different throughput, so we only compare concurrency cells whose output
  hashes match.

**3. For the mapped / file-backed PLE lane.**
- A fresh start leaves almost all of the table out of the page cache (loading the weights evicts it). A decode
  step's rows are known only after the previous step samples, so the host prefetch has no lead, and the GPU faults
  ~30–57 pages per step itself, one at a time.
- Readahead for every page of the step (`MADV_WILLNEED`, then `MADV_POPULATE_READ` per page, both via ctypes) took
  **2.35–2.91 ms/step** off that cold window here. That is 12/12 paired requests faster, warm unchanged, outputs
  identical. A gated wait with a C helper, which we tried first, was slower.
- It's pure Python: [vllm#58835](https://github.com/vllm-project/vllm/pull/58835), on top of our
  checkpoint-mapped backend [vllm#58439](https://github.com/vllm-project/vllm/pull/58439). Your ATS read should have
  the same cold window.

**4. The draft head, next to your reduced draft vocab.**
- We slice the draft vocab too (32k), and then quantize the sliced head rows to **NVFP4**: 606 → 45 MiB read per draft
  step.
- Result: c=1 −3.4 % ms/tok, c=4 +2.8 % tok/s, acceptance slightly up, c=1 output hashes identical
  ([finding 234](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b2c08596f8d624916beac497461c9dfccf9ab640/notes/prefill-investigation.md#L3886)).

**5. GDN RecoverSSM** (you already pin the V2 model runner, which it needs).
- It replaces the native path's per-draft recurrent-state snapshots with a verify-from-checkpoint and one commit
  after sampling. We measured it in `align` mode with prefix caching:
  - agent loop **−16 % per turn**;
  - **+37 % KV tokens** in the same memory;
  - −2.4…−5.2 % per verify cycle.
- Output differs from the native path by the size of a summation-order change.
- Numbers and code are in [bilikaz/qwen38-flash-next-recipe#1](https://github.com/bilikaz/qwen38-flash-next-recipe/issues/1)
  and [vllm#56466](https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579).

And agreed on @Nipale-ai's code-vs-prose split. All our decode probes had been prose essays, which likely understated
MTP depth on code; we are adding a code prompt set before re-measuring k.

*Measured on one GB10 (sm_121), vLLM nightly `1ea7c63f4` + our overlay, with AI assistance (Claude); numbers are
from the linked data.*
