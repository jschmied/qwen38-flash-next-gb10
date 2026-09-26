POSTED 2026-09-26 as https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/19#issuecomment-5846690586, EDITED the same day (user: "answer stays, other stuff goes to new issue"): this is the live text; items 3-5 of the first version moved to issue #78 (issue-miaai-v030-lane-ideas.md).

The two things we owed this thread since 09-09.

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

(Unrelated results for the v0.30 lane, first posted here, are now in their own issue: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/78.)

*Measured on one GB10 (sm_121), vLLM nightly `1ea7c63f4` + our overlay, with AI assistance (Claude); numbers are
from the linked data. Edited 2026-09-26: the off-topic parts moved to the issue above.*
