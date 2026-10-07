DRAFT — user's go 2026-10-07 ("ok": issue 0 + PR 1 together, once PR 1's GPU receipt is in). ashhart/TensorFold issue.
Title: Zig CUDA: Kolibri-1 (kolibri1), block-FP8

This work is underway; please don't start a second Zig port of this family.

Kolibri-1 (`model_type: kolibri1`, 50 layers of 2560, 48 query and 4 KV heads of 128, 384 routed experts of width 512 plus a shared one, top 6 by unnormalised sigmoid weights, sliding window 513 with every fifth layer full attention) ships as block FP8 (128 x 128 e4m3, fp32 `weight_scale_inv`). We serve it on our fork's Python engine on a GB10, with an EAGLE-3 drafter (`josch15366/Kolibri-1-EAGLE3-drafter`).

The Zig CUDA engine has no FP8 path yet, so the plan is small PRs in this order, each with its receipt:

1. Block-FP8 projections: the Python lane matmul's FP8G device code (`nvfp4/qmmf.cu`) as `zig/kernels/cuda/fp8_lane.cu`, a launcher with the Python host's tile, slice and cluster choice, and a `tf-cuda-test fp8-lane` oracle. Model-free.
2. Block-FP8 experts: `fp8/experts.cu` the same way. Model-free.
3. The `kolibri1` family from the command line, drafts off: tokens equal to the Python engine's.
4. The family served through `tensorfold-native`, as an entry in the CUDA registry.
5. The drafter behind `Drafted` (#455), once 4 and #455 are in.

Branch: `jschmied/TensorFold:kolibri1-zig`. 1 is up as a pull request with its receipt.
