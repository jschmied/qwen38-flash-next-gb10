DRAFT — user go given ("ok"). GitHub vllm-project/vllm issue #54521, reply to mmastrac (2026-09-08).

@mmastrac Your `validate_tool_names=True` finding is worth separating from the rest before anyone
chases it, because it is a *reporting* bug and it will hide whatever else you have: a tool call whose
name is not in the request's list produces zero deltas and a `stop` finish with no content and no
`tool_calls`, so a corrupted tool name looks identical to "the model chose not to call a tool". That
is exactly the shape of failure that makes an underlying nondeterminism invisible in a harness. Anyone
debugging tool-call loss on any of these models should set it False first.

On the "random corruption during successful generation" half, a cheap discriminator, since you have a
TP=4 cluster free and I only have one box.

The QSA sparse-attention indexer's top-k hands out output slots by thread arrival, so identical
requests at temperature 0 can select the same keys in a different **order**; the attention then sums
them in that order and greedy decoding forks. On my GB10 the stock kernel fails to reproduce its own
output on **56 of 56** shapes. If that is your mechanism, it is per-GPU and should appear at TP=1 too.

```
# on one Spark, TP=1, temperature 0, prefix caching off:
# send the same prompt 8x and hash the completions
```

If the eight completions fall into more than one class at TP=1, it is this and
[#55122](https://github.com/vllm-project/vllm/pull/55122) is the fix (build it and set
`VLLM_QSA_DET_TOPK=1`, or just diff against a build with it). **If TP=1 is clean and only TP=4
corrupts, it is not this** — it would point at the collective or the draft-KV grouping instead, and
#55506 (mamba spec-decode block tables indexed by batch row rather than request slot) is the closest
open candidate, though that one needs PP≥2 rather than TP.

Two things I cannot help with directly, stated so you do not read my silence as agreement: every
number I have is TP=1 on a single GB10, so I have no evidence about the TP=4 path, and GLM-5.3-Flash
is not a model I run — I am on Qwen3.8-Flash-Next, which shares the indexer but not the parser.

If you do run the TP=1 hash check, the result is useful either way, and I would like to know which way
it goes.
