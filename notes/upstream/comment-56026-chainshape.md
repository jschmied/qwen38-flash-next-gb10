DRAFT — needs the user's go. vllm-project/vllm PR #56026 (2026-09-20).

Independent data on GB10 that supports your 0-of-816,343, and isolates *why* it is 0: on this box the
chain shape is the variable, not the prefix sharing. I set out to check whether your figure
reproduced here, read my own first result as a non-reproduction, and was wrong — your note about the
first request extending a prefix is exactly what my data shows.

**Box**: DGX Spark, GB10, sm_121, aarch64, TP1. vLLM `0.28.1rc1.dev524+g5db652225` **without** your
patch, plus the #53899 PLE-offload port. `Qwen4ExpForConditionalGeneration`, NVFP4 + FP8 head, MTP
`num_speculative_tokens=3`, prefix caching on, resolved attention block **1600**, 4 Mamba groups,
`_warn_if_unannotated_eagle_mamba` firing. `--max-model-len 32768`, so the prompts are far smaller
than your 34K/140K/250K.

Two shapes, same engine, one boot, fresh tag per shape so each starts against a cold region.
Read from `vllm:prefix_cache_hits_total` deltas — `cached_tokens` is inert here (`None` on requests
with ~30,000 cache queries).

| shape | prompt | tokens | hits / queries | rate |
|---|---|---:|---:|---:|
| **chained** (each extends the previous) | 0 writes | 24,459 | 0 / 24,459 | 0.0 % |
| | 1 extends | 30,755 | **0 / 30,755** | **0.0 %** |
| **shared** (all begin with one fixed prefix) | 0 writes | 24,459 | 0 / 24,459 | 0.0 % |
| | 1 first sharer | 27,659 | **0 / 27,659** | **0.0 %** |
| | 2 second sharer | 30,859 | **19,200 / 30,859** | **62.2 %** |

19,200 is exactly 12 blocks of 1,600. Repeated on an identical-prompt probe: 0 on the first
repetition, 6,400 = exactly 4 blocks on the second, and that pattern reproduced across a cold restart
(0 / 0 / 9,600 twice, deterministic).

**So the reading your PR already states is what happens here**: "the first request extending a prefix
still misses with the trailing-block drop on". In a *chained* replay every prompt is a first-extender,
so every one misses and the total is 0 — which is your 816,343. With a *fixed* shared prefix, prompts
from the second consumer onward hit. Nothing about that contradicts the annotation bug; it means the
headline 0 is produced by the chain shape meeting the trailing-block drop.

**The question that leaves, and the reason I am posting rather than just agreeing.** Your "after"
numbers were taken with `disable_eagle_block_drop` as a stand-in for #52244. If the chained 0 is the
trailing-block drop rather than the annotation, then the annotation fix's own contribution to *that*
replay is separable from the drop's. Do you have an arm with your patch applied and the block drop
left **on**? On this box that would distinguish them, and I am happy to run it: our launcher takes
`disable_eagle_block_drop`, so the 2x2 is four boots at about 12 minutes each.

**Not measured**: your patch (this is the unpatched arm only), the `disable_eagle_block_drop` arm,
anything above 32,768 tokens of context, and the chained shape beyond two prompts — our context
budget stopped it where yours runs to 250K. One boot per shape, not three.

_AI assistance (Claude Code) was used for this analysis; every number was checked against the run that
produced it._
