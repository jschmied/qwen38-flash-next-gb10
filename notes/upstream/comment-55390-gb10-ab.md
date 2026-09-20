DRAFT — needs the user's go. vllm-project/vllm PR #55390 (2026-09-20).

Tested this PR on GB10 with a two-arm A/B. It does what it says — the warning stops — but on this
model and this probe it changes no measured behaviour, because **cross-request reuse was never
disabled here**.

**Box**: NVIDIA DGX Spark, GB10, sm_121, aarch64, TP1, 121.6 GiB unified. vLLM
`0.28.1rc1.dev524+g5db652225` (base 2026-09-08) plus the #53899 PLE-offload port.
`Qwen4ExpForConditionalGeneration`, locally derived FP8-head NVFP4 checkpoint, `--max-model-len
32768`, `--max-num-seqs 16`, MTP `num_speculative_tokens=3`, prefix caching on, resolved attention
block **1600**, 4 Mamba groups.

Arms differ **only** in `vllm/v1/core/kv_cache_utils.py`, each on its own boot, journal scoped to
that boot's start:

| arm | warning | GPU KV | rep2 hits / queries | rate | wall cold / rep1 / rep2 |
|---|---|---|---:|---:|---|
| unpatched `dev524` | **2** | 578,901 | 19,200 / 26,910 | **71.3 %** | 3.66 / 3.45 / 1.19 s |
| + this PR | **0** | 547,693 | 19,200 / 26,919 | **71.3 %** | 3.61 / 3.42 / 1.18 s |

3 prompts x 3 passes per arm, ~8,973 prompt tokens each, read from `vllm:prefix_cache_hits_total` /
`_queries_total` deltas. Hit counts are identical between arms and wall times differ within noise.
The KV difference is not attributable: across five boots today the pool ranged 547k-579k.

**Why reuse is not disabled here, in the arithmetic.** 8,973 tokens is 5.61 blocks of 1600. The hit
is 6,400 = exactly **4 full blocks**. The 2,573 unhit tokens are the 973-token partial tail **plus
exactly one full block** — i.e. the EAGLE volatile trailing-block drop, doing precisely what it is
meant to do. Reuse is reduced by one block, not disabled.

**Two things worth separating, because I nearly conflated them.**

1. **Ours is not the case [#56026](https://github.com/vllm-project/vllm/issues/56026) reports.**
   That one replays *different* prompts sharing a prefix (34K/41K/45K chained, 140K then 250K) on
   this same model and gets 0 hits from 816,343 queried. We re-ask an **identical** prompt, where
   every block is shared. Our result does not refute theirs — it bounds the claim: the flag-all
   fallback does not disable all cross-request reuse on `qwen4_exp`, at least not the
   identical-prefix case. If it would help, say which prompt shape you want and I will run it.
2. **The alarming text we saw is stale.** Our base predates
   [#56791](https://github.com/vllm-project/vllm/pull/56791) (merged 2026-09-14), so this build
   still says "prefix-cache reuse across requests will be disabled ... store without ever serving a
   hit". On the trimmed message our observation reduces to what
   [#55519](https://github.com/vllm-project/vllm/pull/55519) already states with more arms — the
   warning fires in runs where reuse is plainly working. Posting anyway because the A/B isolates
   *this* PR rather than the warning, and because a GB10 + `qwen4_exp` arm did not seem to be in the
   thread yet.

**Not measured, and it matters**: **one boot per arm**, not our usual three, so a between-boot effect
is not excluded — the hit counts being bit-identical (19,200 both) is the reason I think that is
safe, not a separate control. Shared-prefix-across-different-prompts is untested. `disable_eagle_block_drop`
is untested here, though it is the lever that should recover the dropped block.

Happy to re-run any of the above on this hardware; reproducing costs about 12 minutes a boot.

_AI assistance (Claude Code) was used for this analysis; every number was checked against the run
that produced it._
