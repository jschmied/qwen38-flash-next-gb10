DRAFT — needs the user's go. ashhart/TensorFold PR #212 comment (2026-10-02).

Reproduced on a second GB10 (DGX Spark, sm_121), Flash Next EXL3 3.05 bpw, one process an arm, three alternating
rounds ([data](https://github.com/jschmied/qwen38-flash-next-gb10/tree/63b995d2c939e7500a8b2ea7b8499fb172d0ddec/notes/data/tfexl3/pr212)):

| prefill (s) | v0.6.1 | #212 | #184 → #207 |
|---|---|---|---|
| 8k | 11.39 / 11.42 / 11.44 | 5.30 / 5.31 / 5.29 (−53.5…−53.8 %) | 8.58–8.67 (−24 %) |
| 32k | 46.23–46.87 | 21.29–21.35 (−53.8…−54.6 %) | 34.69–36.26 (−23 %) |

The first 16 tokens after both prompts match across all three. Decode with drafts (8 prompts x 256 tokens) is level:
71.8–72.0 vs 71.9 tok/s, same rounds and drafts. Your tests and ours (test_exl3_*, test_qwen4_exp_exl3,
test_group_kernel_smem): 90 passed.

This supersedes my prompt-side PRs, so I'll close #184, #195 and #207. #191 and #193 were only measured on prompts; I
haven't shown they help decode windows, so I'll close them too unless that turns out otherwise.

Written with AI assistance (Claude Code).
