Draft for #548's second step, as a concrete example of the shape it proposes: the expert plan that groups routed pairs by expert (`experts.route` on `experts.cu`'s plan kernels) moves out of `families/nemotron` into `zig/src/cuda/grouped.zig`, and Nemotron routes through it. Behaviour doesn't change: same kernels, same grids, same arguments.

- `cuda/grouped.zig`: `maxItems`, `Plan` (the scratch the plan writes) and `Router` (`resolve` from the loaded experts module, `route` = the one-block plan up to 1,024 pairs, else rank, offsets and scatter).
- `families/nemotron/cuda_kernels.zig`: `Plan` and `maxItems` become aliases, `Kernels` holds a `Router` instead of four plan functions, and `Ops.plan` delegates. No caller changes. The family is 42 lines shorter.
- `tf-cuda-test grouped-plan <dir>` with `oracle/grouped_plan.py`: members, items and counts against `experts.route` (python-0.6) for one-block and wide plans at tiles 16 and 64 (1 to 2,048 rows, 7 and 9 slots, 33 and 385 experts).

Every MoE port needs this plan: Kolibri's FP8 experts (#481), Flash Next's NVFP4 experts (#472) and GLM's EXL3 experts (#509). With it here, the per-format expert kernels can share one plan instead of each carrying its own.

Receipts so far: `zig build test` passes on GB10 (Linux aarch64), the CUDA kernels build for sm_121, and `lean_check` is clean for the touched files.

Still to come before this leaves draft:
- the `grouped-plan` oracle test on GB10;
- Nemotron 3.5 Lightning greedy and sampled replies equal before and after this change on GB10, at the revision #542's receipt used.

If you'd rather keep the plan inside the families, say so and I'll close this.
