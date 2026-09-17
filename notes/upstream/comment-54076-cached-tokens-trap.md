Two things before I run the arm you asked for: a measurement trap that would invalidate that exact cell, and an existing result that corroborates the scoping commit from a different direction.

**The cell you asked for cannot be read from `cached_tokens`.** We measured this on 2026-08-24/25 while verifying #53479 (Qwen3.8-27B-NVFP4, MTP n=3, fp8 KV, align mode): `usage.prompt_tokens_details.cached_tokens` returned **0 on every request, including requests that provably hit**. A first full pass reported "never hit" on both arms and would have falsified a correct prediction. What works:

- `vllm:prefix_cache_hits_total` deltas per request, client-side, no patching; and/or
- instrumented hit logging in `HybridKVCacheCoordinator.find_longest_cache_hit`.

The two agreed exactly (base total 8,000 = 4,800 + 3,200). Flagging it now because anyone reading "cached tokens on an immediate same-prompt re-ask" literally will get zeros on both arms and conclude the fix is a no-op.

**On the scoping commit, we have the flag-shaped version of the same result.** On a main-build Flash-Next serve with MTP n=3 and prefix caching on, `disable_eagle_block_drop` (#53388) gave, over 3 starts:

| | without | with |
|---|---|---|
| cached tokens per warm turn | 4,800 | 6,400 |
| warm agent turn | 2.05 s | 1.52 s |
| MTP acceptance | 53–56 % | 57–60 % |

Same mechanism as your `use_eagle_preserves_target_kv_cache()` predicate, reached by a manual flag instead of a principled one: the trailing full prefix block is dropped on every hit and re-prefilled next turn. Your version is better, because it decides from the drafter's actual KV behaviour rather than requiring the operator to know to set a flag.

Two caveats on our figure. It is `mtp`, not dflash/dspark, so it is adjacent to your case rather than identical. And it does not address the first-repetition cold turn in align mode, which the flag cannot touch.

I will run the patched-vs-unpatched arm you asked for — `prefix_cache_hits_total` deltas, EOS-correct harness, 3 starts — and add the third arm with the scoping commit. The PR is Python-only, so it overlays onto an existing build without a rebuild. One caveat: the machine is single-GPU and currently committed to another model, so this is serialised rather than immediate.

*AI assistance was used in preparing this comment; the measurements referenced are ours and were reviewed before posting.*
