Two things that may still be useful, and an apology: **I cannot run the arm I offered.**

**First, the apology.** I offered the patched-vs-unpatched measurement and you accepted it. Since then this machine has been reassigned to a different model, and the Qwen3.8-Flash-Next checkpoint is no longer on it — only a metadata stub remains. Restoring it is a ~100 GB pull that I am not going to make for this, so the cell you asked for will not come from me. I should not have offered without checking that the model was still resident. Apologies for the delay this has cost you.

The two things below stand on measurements already taken, so they are worth posting regardless.

**The cell you asked for cannot be read from `cached_tokens`.** We measured this on 2026-08-24/25 while verifying #53479 (Qwen3.8-27B-NVFP4, MTP n=3, fp8 KV, align mode): `usage.prompt_tokens_details.cached_tokens` returned **0 on every request, including requests that provably hit**. A first full pass reported "never hit" on both arms and would have falsified a correct prediction. What works:

- `vllm:prefix_cache_hits_total` deltas per request, client-side, no patching; and/or
- instrumented hit logging in `HybridKVCacheCoordinator.find_longest_cache_hit`.

The two agreed exactly (base total 8,000 = 4,800 + 3,200). Worth flagging because anyone reading "cached tokens on an immediate same-prompt re-ask" literally will get zeros on both arms and conclude the fix is a no-op.

**On the scoping commit, we have the flag-shaped version of the same result.** On a main-build Flash-Next serve with MTP n=3 and prefix caching on, `disable_eagle_block_drop` (#53388) gave, over 3 starts:

| | without | with |
|---|---|---|
| cached tokens per warm turn | 4,800 | 6,400 |
| warm agent turn | 2.05 s | 1.52 s |
| MTP acceptance | 53–56 % | 57–60 % |

Same mechanism as your `use_eagle_preserves_target_kv_cache()` predicate, reached by a manual flag rather than a principled one: the trailing full prefix block is dropped on every hit and re-prefilled next turn. Your version is better, because it decides from the drafter's actual KV behaviour instead of requiring the operator to know to set a flag.

Two caveats. It is `mtp`, not dflash/dspark, so it is adjacent to your case rather than identical. And it does not address the first-repetition cold turn in align mode, which the flag cannot touch.

*AI assistance was used in preparing this comment; the measurements referenced are ours and were reviewed before posting.*
