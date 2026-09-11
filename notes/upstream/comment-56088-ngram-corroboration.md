Independent confirmation of blocker 1, from a different direction: **ngram speculation**, not a DFlash/DSpark drafter, on **1x DGX Spark (GB10, sm_121, TP1)**.

```
WARNING  Model Runner V2 does not yet support ngram/ngram_gpu; using the V1 model runner instead.
RuntimeError: PLE inputs were not prepared
```

Same error, same guard (`nvidia/model.py:300`). So this is not specific to the drafter — **any method that forces the V1 runner cannot serve this target**, and `ngram`/`ngram_gpu` are on V2's unsupported list, so they force it by construction.

One thing worth knowing for whoever fixes it: **blocker 2 sits directly behind blocker 1.** Patching the V1 PLE plumbing gets you to the QSA ring assert next — `capacity 12 must divide block size 1616` with `num_speculative_tokens=5`. That is [#54552](https://github.com/vllm-project/vllm/issues/54552), and the fix is already written in [#54912](https://github.com/vllm-project/vllm/pull/54912) (open, awaiting review). I confirmed the ordering by patching the ring locally: the assert cleared on all 12 QSA layers, and the PLE error is what surfaced immediately after.

No opinion on the design question in item 1 — that is the maintainers' call, and it is why you held the PR.
