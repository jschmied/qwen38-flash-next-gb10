SUPERSEDED by PR #211 — not posted. ashhart/TensorFold #173 (closed) comment (2026-10-02).

Thanks. As far as I can read 0.6.1, `--precision checkpoint` reaches the 27B only: `precision.mode()` and
`tensorfold.cuda.nvfp4.checkpoint` are used by `families/qwen3_5` alone, and Flash Next's routed NVFP4 experts still
run `nvfp4_expert_kernel` on bf16 rows (`cuda/nvfp4/experts.cu`), whatever the flag. That path, the grouped routed
experts at 36.7 % of an 8k prompt, is the one this issue asked about.

Is FP4 x FP4 for Flash Next's routed experts on your list, or would you take it as a PR on the `--precision checkpoint`
contract (grouped, rows quantized under each expert's static input scale, prompts and decode)?

Written with AI assistance (Claude Code).
