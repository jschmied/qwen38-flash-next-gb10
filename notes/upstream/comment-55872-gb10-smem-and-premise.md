DRAFT — user go "post". vllm-project/vllm PR #55872, comment (2026-09-09).

@jahnclawdmonet's shared-memory finding also explains a failure we reported earlier without a cause, and
we have two measurements since that bear on the PR's premise.

**Third GB10, and the mechanism names our earlier symptom.** On 2026-09-08 we reported on #55122 that
this backend does not start on sm_121 — `TopKRaggedTransform failed with error code operation not
supported` at engine init, on a **single** GB10 at TP=1, with the `native` arm on the same build fine.
We could not say why. @jahnclawdmonet's number accounts for it: this device reports

```
NVIDIA GB10, sm_121, 48 SMs
shared memory per SM: 102400 bytes      (FlashInfer 0.6.18 requires >= 131072)
```

So the failure is not TP- or Ray-specific — TP=1 and TP=2 fail for the same structural reason, on three
GB10s across at least three driver versions. Anything gated on 131072 bytes/SM is unavailable on this
whole hardware family, not just misconfigured on it.

**On the premise, and we say this about our own PR first.** The motivation here is that "index selection
can be non-deterministic when values tie at the TopK boundary". On this workload we can no longer find
that tie. Two runs on Qwen3.8-Flash-Next NVFP4, GB10, TP=1, MTP-3, k = 512 blocks:

1. **The boundary does not tie.** A census inside `_topk` over 400 instrumented calls: 87,257 rows, of
   which **6,192 performed a real selection** (the rest had fewer visible blocks than the budget, so
   everything was taken). Rows with *any* value equal to the k-th: **0**. Not "tied but resolved
   consistently" — never tied. Exact ties between float32 logits out of a real GEMM are simply rare.

2. **The divergence is in the scores, not the selection.** Hashing the *input scores* and the *selected
   index set* per call across 7 byte-identical greedy requests: on a stock server, all **13** comparable
   prefill calls have **different input scores**, and **zero** have identical scores with a different
   selection. With our four determinism fixes on, all 13 are bit-identical on both. Boundary gaps
   (kth − (k+1)th) are min 4.58e-05, median 3.20e-04, so a perturbation of that order is enough to
   reorder the selection — which is @rybruscoe's reading on #54521, and it puts the cause upstream of
   the selection kernel.

**We published exactly this about our own #55122 before writing it here**, and we think it applies the
same way: a deterministic tie-break is *kernel correctness under ties*, not a route to end-to-end
reproducibility on traffic where the boundary never ties. That does not make this PR wrong — your body
already declines to claim a quality or performance improvement, and correctness under a rare condition
is worth having. It does mean a user adopting it for reproducibility on this model would not get it, and
@jahnclawdmonet's native-backend result is consistent with that: still 3–6 distinct completions of 8
above the budget, clean below it.

**Bounds, so this is not read as more than it is.** Both runs are prefill-only comparisons on one model
on one GPU family. Decode calls could not be aligned at all in the stock arm, because once outputs
diverge the MTP acceptance changes and so does the number of decode steps — that drift is itself a
symptom rather than a nuisance. And a tie rate indistinguishable from zero at ~6×10³ selecting rows is
not zero at 10⁶; a long-context, low-entropy workload might sample the boundary very differently.

TP=1 is the axis this thread does not otherwise have, and the box is free — happy to run any specific
cell you want, including a census on a workload you think *should* tie.

Data: [`notes/determinism-investigation.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/determinism-investigation.md) findings det-190 and det-191.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
