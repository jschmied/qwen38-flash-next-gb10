POSTED as #207 (2026-10-02). ashhart/TensorFold PR, base pr-141-0.6.1, head jschmied:exl3-stack-win (2026-10-01).
Title: Flash Next CUDA EXL3: routed-expert windows of 2,048 rows, one call a prompt chunk (prompts −4…−6 %)

Stacked on #195 (#184 → #191 → #193 → #195); the new commit is the last one, one line.

`MOE_WINDOW = 1024` was set by the grouping's shared memory ("keeps every pick in 48 KB"). Since #191 the grouping
holds no picks in shared memory, so that limit is gone, and a 2,048-row prompt chunk now runs its routed experts in
one call instead of two: half the grouping and launches a chunk, and fuller 16-row member tiles. Decode windows are
unchanged (they were already one window).

Cost: the fixed routed-expert scratch doubles. On Flash Next (11 slots, hidden 2,560, expert width 640) the geometry's
reservation goes from 454 to 908 MiB, taken from the cache budget.

## Measured

One GB10, Flash Next EXL3 3.05 bpw, three alternating rounds, prefill seconds
([data](https://github.com/jschmied/qwen38-flash-next-gb10/blob/b35c27c7973addc787434929ab5a6f694904485a/notes/data/tfexl3/d4.txt)):

| | #195 | this PR | per round |
|---|---|---|---|
| 8k prompt | 8.94 / 8.93 / 8.96 | 8.63 / 8.58 / 8.61 | −3.5 / −3.9 / −4.0 % |
| 32k prompt | 36.13 / 37.28 / 37.06 | 34.85 / 34.83 / 36.19 | −3.5 / −6.6 / −2.3 % |

The first 16 tokens after both prompts are identical in every run.

## Tests

On GB10: test_exl3_experts, test_qwen4_exp_exl3 and test_group_kernel_smem: 78 passed, 52 skipped.

Written with AI assistance (Claude Code); the author reviewed every change.
