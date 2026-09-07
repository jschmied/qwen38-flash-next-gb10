DRAFT — user go given ("yes, and comment on overlapping"). GitHub vllm-project/vllm PR #55314 (2026-09-07).

@Dovis01 We are fixing the same defect from opposite ends and had not found each other — #55122 has
been open on this kernel since before this PR, and #53287 is a third. Cross-linking so all three do
not stall in parallel, and because I think they are more complementary than competing.

**Where we agree.** Same root cause: candidates past stash capacity dropped in arrival order, so the
selected *set* diverges from `torch.topk` on tied or tightly clustered rows. Your descend-the-key-bytes
approach fixes that while keeping the buffers; ours removes the buffers and rescans per key byte. Both
give an exact set.

**Where they differ, and it is not a style question.** This PR keeps `atomicAdd` for the output slot:

```
const int pos        = atomicAdd(&decode_smem[sOUT_abs], 1);
const int output_pos = atomicAdd(&shared_output_count, 1);
```

so the *order* of the emitted indices still depends on thread arrival. That matters here because the
sparse attention sums the selected keys in output order, so an order-only change still forks greedy
decoding between identical requests — which is the bug #54521 reports and what #55122 exists to fix.
Your clip "after full-key equality, where any subset is a valid selection" is sound for exactness and
not sufficient for reproducibility: *which* valid subset you get can still vary run to run.

Two checkable consequences, offered as a suggestion rather than a criticism:

1. Your three new tests are all `..._oversized_threshold_bin` — they compare against `torch.topk`
   once. **Calling the same kernel 6× on one input and requiring bit-identical output** would show
   this directly; on our GB10 the unmodified kernel reproduces its own output on **0 of 56** shapes.
2. If you want the order property, the cheap version is a packed `BlockScan` over the (greater, equal)
   flags instead of the atomic — it costs one scan and removes the arrival dependence entirely.

**What you cover that we deliberately do not.** `cooperative_topk` and `topk_histogram_4096`. #55122
bypasses the 2048-bin path rather than fixing it, and `cooperative_topk` we cannot even exercise —
all 51 of its cases fail on sm_121 with `cooperative_topk launch failed: invalid argument`
(cluster launch rejected on GB10). Your PR is the only one touching those.

Useful pointer from your side that I had missed: the SGLang origin
([sglang#37625](https://github.com/sgl-project/sglang/pull/37625)) and that vLLM's stash stores were
already capacity-guarded, so only the selection divergence was live here. That is a cleaner statement
of the scope than mine.

Happy to run your branch on a GB10 (sm_121) against our determinism harness — 56 shapes, 6 calls each,
exact-reference comparison — if that would help. It would answer the set question and the order
question in one pass.
