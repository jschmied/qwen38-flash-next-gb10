Tested it. **#55872's FlashInfer backend does not start on sm_121.**

GB10 / sm_121a, aarch64, vLLM `0.28.1rc1.dev524+g5db652225`, `Qwen3.8-Flash-Next` NVFP4, TP1.
Your patch applies cleanly to that tree (7 files, 0 failed hunks), `sparse_attn_topk` imports, and
the API it calls is already present in **flashinfer 0.6.17** — no 0.6.18 bump needed. But with
`--dsa-topk-backend flashinfer --dsa-topk-tie-break small` the engine dies during init:

```
RuntimeError: Check failed: (status == cudaSuccess) is false:
  TopKRaggedTransform failed with error code operation not supported
```

The `native` arm on the *same patched build* starts fine, so this is the backend rather than your
patch or anything local to me. `operation not supported` from a CUDA call at init reads like no
sm_121 implementation behind `top_k_ragged_transform`. Happy to run any diagnostic you want on this
box — I appear to be the only GB10 in this thread, and I'd rather help you fix it than use it as an
argument.

**On accuracy you are right, and I should say so plainly.** #53287's conclusion holds, and my own
evidence agrees with it: in @k3dani's independent 50-item KIE suite the *stock* kernel scored
**higher** (97/100 vs 95/100). I have no evidence of an accuracy regression from this
non-determinism and I am not claiming one.

What I have is **reproducibility**, which is a different axis: the same suite showed stock at
**13/50 unstable items vs 0/50**, and #54521 opened on user-visible text corruption (dropped and
transposed Thai tone marks) rather than on a benchmark delta. For anyone diffing outputs across runs
or bisecting a regression, "the same request returns a different answer" is the defect, even when
average quality is unchanged.

**So I'll drop the part you object to.** Changing the default is not what I need, and an opt-in
backend is a reasonable shape. What I'd suggest instead is that your PR is the config surface and
mine is one implementation behind it: `--dsa-topk-backend` selects, and on hardware where the
FlashInfer path is unavailable the deterministic native kernel can be what the flag selects. That
gets reproducibility to users on both, and neither of us has to win.

**On the performance objection** — that was fair, and I've now addressed it rather than argued with
it. Two commits pushed today, both correctness-gated (`FAILS: 0`) before any timing:

- a blocked 4-item emission (one `BlockScan` + barrier per 4,096 elements instead of per 1,024);
- `RADIX_THRESHOLD` 16,384 → 22,016, the shared-memory caching bound.

Measured on GB10, interleaved arms, 3 starts each. On a 48-cell grid the two compose (0 cells where
both together are worse than either alone), the worst cell goes **2.13× → 1.78×**, and the kernel is
**at or below stock on 27 of 43 cells** where the stock control is tight. The 1.3–4.3× table in the
PR body predates these and I am updating it.

One small thing for your PR regardless of the above: your `qsa_indexer.py` hunk touches the same file
a local determinism overlay of mine patches. It applied cleanly here, so this is a note rather than a
problem, but it may be worth knowing if someone reports a conflict.

<!-- AI disclosure: this analysis was produced with AI assistance; I reviewed every line and ran every number quoted. -->
