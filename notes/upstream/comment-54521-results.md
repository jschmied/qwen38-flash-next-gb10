# DRAFT — results follow-up on vllm-project/vllm#54521 (post on go)

Position-resolved regression on sm_121 as requested (main dev401, no spec, prefix cache off, batch 16384, temperature 0,
`prompt_logprobs=5`, 8 sequential + 8 concurrent identical requests, 64-token greedy hashes; 1,460 / 1,999 / 5,960-token
prompts — below, at and well above the 2,048 budget):

| arm | prompt | seq: first divergent pos / top-1 flips / mean spread | conc flips | 64-tok distinct/8 |
| --- | --- | --- | --- | --- |
| stock (2 starts) | 1,460 | 1 / 924–988 / 6.4–6.9 | 896–954 | 8 |
| stock | 1,999 | 1 / 1,042–1,069 / 5.8–5.9 | 1,179–1,258 | 7–8 |
| stock | 5,960 | 1 / 2,974 / 4.9 | 3,391 | 8 |
| det finalize (`use_fused_finalize=False`) | 1,460 / 1,999 / 5,960 | 1 / 758 / 5.3 · 1 / 788 / 4.2 · 1 / 2,660 / 4.5 | 693 · 1,237 · 3,215 | 4 · 2 · 5 |
| det finalize + det `persistent_topk` | 1,460 / 1,999 / 5,960 | 1 / 758 / 5.3 · 1 / 788 / 4.2 · 1 / 2,253 / 3.9 | 693 · 1,237 · 3,101 | 4 · 2 · 3 |

Two things in there. Above the budget the deterministic top-k removes a further 15 % of the flips and 2 of 5 distinct
completions — that part is #55122. Below the budget the two det arms are identical to the digit across server starts, so
the remaining "divergence from position 1" is not noise at all: with both fixes the forward is bit-exact in eager mode and
with `cudagraph_mode=NONE` at every length (0 flips, spread 0.000, 1/8 distinct), and 16 identical sequential requests give
one class. With graphs enabled they give two classes — the cold first request, then fifteen bit-identical ones — because
of a separate defect in the PLE CPU-offload branch this model needs on GB10: every forward consumes the previous step's
PLE outputs (semaphore one step ahead after `capture_model()`'s dummy signal). Details and fix on #53899 / https://github.com/peakcrosser7/vllm/pull/13; it is
not a property of the indexer or the MoE kernels. So for the regression set here: fix `persistent_topk` (#55122) for the
above-budget regime, `use_fused_finalize=False` for the MoE, and the offload semaphore for everything else; with all three
on sm_121 every real step consumes exactly its own rows from the first one (32/16/2/1 at init, then 1,460 ×3, 1,999 ×2; hashes equal to the NONE run's), the cold first request gives the warm logprob (−0.2638), 16 identical requests = 1 class, and the position-resolved set is bit-exact sequentially at 1,460 / 1,999 / 5,960 tokens (0 flips, spread 0.000, 1/8 distinct 64-token completions each); the concurrent batches keep 0 / 416 / 665 flips, identical to the cudagraph-off run — the batch-shape axis, not this defect.
