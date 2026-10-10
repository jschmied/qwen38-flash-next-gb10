POSTED 2026-10-10 as TF #548 comment (user: "yes, a and b"). https://github.com/ashhart/TensorFold/issues/548#issuecomment-6101684153

Claiming item 4 on CUDA, the shared layer pieces, driven by Kolibri 1 (#481) as their first user: RMSNorm and the fused add + RMSNorm, QK norms with RoPE and the cache write, the sigmoid router with bias and its top-k, the experts' weighted combine, then sliding-window and tree attention for decode and verify rows.

Each one is our own CUDA kernel in `cuda/`, checked against a float64 or host reference inside its test (as in #633), with a row's bits the same at every row count, and drafted == plain on the family that uses it. They go up as small PRs, simplest first (the norms, combine and top-k). Nemotron's own glue stays as it is unless you'd like it moved onto them.

This is CUDA only. #617 covers the same operations on HIP, and the shapes and contracts will match where they can.
