POSTED 2026-10-10 as TF #548 comment (user: "claim and continue"). https://github.com/ashhart/TensorFold/issues/548#issuecomment-6093874349

Claiming the quantized-weight interface, next after #558 as you said.

Plan, as small PRs in this order:

1. **`cuda/qlinear.zig`: one weight view and one dispatch.** A tagged weight (`affine4`, `fp8g`, more formats later) with
   decode-row and prompt-row entry points. The first PR moves Nemotron's MLX affine-4 path (`QLinear`, the lane_gemv /
   split / cluster decode paths and `prefill_matmul`) behind it, with Nemotron routed through it and its bits unchanged.
   #482's `fp8.Lane` goes behind the same interface unchanged. Receipts as in #558: the `tf-cuda-test` fixtures, and
   Nemotron replies identical on main and the PR, greedy and sampled.
2. **NVFP4** behind the same interface: the lane matmul for decode rows and a prompt GEMM for prompt rows, each with an
   oracle fixture. It exists on our `cuda-nvfp4` branch, SASS-identical to the Python build's kernels.
3. **Grouped experts per format** (FP8, NVFP4) behind #558's plan.

@Bizuayeu, thanks. A GB10 run of `cuda-nvfp4` and `cuda-nvfp4-experts` with `nvidia/GLM-5.3-Flash-NVFP4` would be
welcome; it would show early whether the weight view fits the ModelOpt layout as stored.
