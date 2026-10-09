POSTED 2026-10-09 ~13:00 as TF #548 (user: "yes, post this list as issue"). https://github.com/ashhart/TensorFold/issues/548

Title: Zig CUDA ports: shared layers below `lanes.Backend`, and one place to coordinate them

Several model families are being ported to the Zig CUDA engine at the same time, and below the lane engine each port is building the same pieces again. The lanes engine, the CUDA runtime, the family registry, the checkpoint loader and the prompt cache are shared. Everything between the runtime and a family's `Backend` currently isn't.

What is being built more than once right now:

| Piece | Built separately in |
| --- | --- |
| Quantized linear layers (block-FP8, NVFP4, EXL3, MLX affine) | #482 and our [`cuda-nvfp4`](https://github.com/jschmied/TensorFold/tree/cuda-nvfp4) branch (FP8 and NVFP4 lane matmul), #496 (EXL3 decoder), the Flash Next port in #472 (MLX affine) |
| Grouped MoE experts and their routing plan | our [`cuda-fp8-experts`](https://github.com/jschmied/TensorFold/tree/cuda-fp8-experts) / [`cuda-nvfp4-experts`](https://github.com/jschmied/TensorFold/tree/cuda-nvfp4-experts) branches (`experts.route`'s plan + expert kernels), Nemotron's own experts, the Flash Next kernels on BobClawblaw's [`flashnext-zig-port`](https://github.com/BobClawblaw/TensorFold/tree/flashnext-zig-port) branch |
| Copying the Python extensions' device code into the Zig build | our generated copies, BobClawblaw's `gen_copies.py` (renaming the namespace or reordering instantiations changed two kernels' SASS) |
| Oracle fixtures and their test harness | `tf-cuda-test` fixtures (#482), #507 (EXL3 fixture generator), the replay harness on `flashnext-zig-port` |
| Norms, RoPE variants, full and sliding attention, routers (softmax, sigmoid + bias), SwiGLU | each family |
| Non-CUDA runtimes | HIP (#463, #490), Level Zero (#522, #523) |

The ports in flight: Flash Next (#472), Kolibri 1 (#481), GLM-5.3-Flash EXL3 (#509), the Qwen3.5 dense family (#503, #520, #528). The same `direct_io` test fix has also arrived three times (#486, #500, #540).

Proposal, in the order that would save the most duplicated work:

1. **One quantized-weight interface.** A weight view per format, with one dispatch for decode rows and prompt rows, and each format's packing next to it. `qmmf.zig` in #482's stack already does this for FP8G and NVFP4.
2. **One grouped-MoE layer.** The routing plan plus per-format expert kernels behind one call. Our plan and FP8/NVFP4 expert kernels are byte-identical to the Python kernels on GB10 and could be the starting point.
3. **One kernel-copy tool and one fixture format.** It should copy device code without changing its SASS, compare the SASS against the Python build, and write fixtures that every family's test command reads.
4. **Shared layer pieces** (norms, RoPE, attention, routers), so a new family is mostly wiring.

Every shared piece would have to keep what the families have now: bit-identical output against the Python engine, and no slower than a specialised kernel. Where a family needs a specialised kernel, it can still register one behind the same interface.

Two questions for @ashhart:

- Is this a direction you want? If so, which of the four first?
- Would one tracking issue (this one, or a new one) work as the place where ports claim a piece before building it?

We can contribute 1 and 2 from what we already have, as small PRs. We'd hold them until you've set the direction.
