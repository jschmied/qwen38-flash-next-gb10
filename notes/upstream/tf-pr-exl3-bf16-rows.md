POSTED 2026-10-01 (user: "yes"). ashhart/TensorFold PR #195, base pr-141-0.6.1, head jschmied:exl3-bf16-rows dbda281, stacked on #193.

Stacked on #193 (#184 → #191 → #193); the new commit is the last one.

A Flash Next prompt keeps its per-slot expert outputs in bf16 (`MoEBuffers.y`), so `_exl3_moe` copied every window's
fp32 `routed()` rows into it. On a GB10 that's 441 `bfloat16_copy_kernel` launches, 3.0 % of an 8k EXL3 prompt. The
down epilogue now stores the caller's dtype directly (`routed(..., y_out=)`): the same fp32 value, rounded to nearest
even as the copy did. So the rows are bit-identical, and the fp32 rows and the copy are gone. Decode windows keep
their fp32 rows.

## Measured

One GB10, Flash Next EXL3 3.05 bpw, two alternating runs:

| | #193 | this PR |
|---|---|---|
| 8k prompt, prefill (s) | 9.62 / 9.61 | 9.19 / 9.09 |
| 32k prompt, prefill (s) | 38.77 / 38.77 | 37.20 / 36.72 |

The first 16 tokens after both prompts are identical.

## Tests

- `tests/cuda/test_exl3_experts.py`: bf16 rows out equal the fp32 rows rounded with `.to(torch.bfloat16)`, and the
  fp32 path is unchanged.
- On GB10: test_exl3_experts, test_qwen4_exp_exl3 and test_group_kernel_smem: 78 passed, 52 skipped.

Written with AI assistance (Claude Code); the author reviewed every change.
