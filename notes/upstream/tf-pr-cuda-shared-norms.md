# TF PR: cuda shared norms + embed rows (#736)

POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/736 (user: "then post it"). Branch fork/cuda-shared-norms c1138e24. Nemotron norm NOT moved (different sum order = bit change); embed16 NOT moved (no ModelOpt Nemotron to run).

## What this changes

Nearly every family needs the same row kernels around its projections: RMSNorm, the residual add with its norm, a gated head norm (the Gated DeltaNet families), and token rows from the embedding table. Today each CUDA family writes its own. Nemotron has `nemotron_norms.cu`, #608's 27B runs Triton for the same steps, and the next family would write them a third time.

This adds them as shared modules, built the way the other `cuda/` modules are: each has a float64 reference in `core`, a `tf-cuda-test` command, the same bits at any row count, and a mutation the test catches.

- `norms.cu` / `cuda/norms.zig`: `rms` (f32, f16 or bf16 in and out, optional weights), the residual step (`add_rms`), `add_rms_h` (a bf16 residual stream: h = x + r, y = rmsnorm(h) · w, plus y's 64-value group sums for a 4-bit projection) and `gated_rms` (rmsnorm(y) · w · silu(z) per head, from strided rows). One block per row, sums in a fixed order. Rows up to 8,192 wide stay in registers.
- `embed_rows.cu` / `cuda/embed_rows.zig`: token rows from a 4-bit affine table (MLX layout, groups of 64) or a bf16 table, 16 bytes per thread.
- `core/norms_ref.zig` and `core/act_ref.zig`: the float64 references.

Nemotron is the first user: its 4-bit embed now runs on `embed_rows`, and `tf_nemo_embed` is removed. It gives the same bits, since both compute `fmaf(q, scale, bias)` per value.

Two things are deliberately left out:
- Nemotron's norms. The shared norm sums in a different order, so moving them would change Nemotron's bits for no measurable gain. That can be its own change if you want one norm kernel.
- Nemotron's bf16 table (`tf_nemo_embed16`, ModelOpt checkpoints). I have no ModelOpt Nemotron checkpoint to run end to end.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus these two commits; one DGX Spark (GB10, sm_121, driver 580.173.02, CUDA 13); `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`.
- Exactness, kernels:
  - `tf-cuda-test norms` and `norm-rows`: every output is within its float64 tolerance (worst 0.996 of a tolerance), and fewer rows (1, 3, 17) give the same bits as the most (300 or 64).
  - The cases cover widths 64 to 9,216 (past 8,192, the path that re-reads its row), Nemotron's 2,688 and the 27B's 5,120, gated heads of 64, 128 and 256, and embed widths 192, 2,688 and 5,120.
  - `tf-cuda-test glue`: 100 of 100 Nemotron glue comparisons byte-equal, the embed now through `embed_rows`.
- Exactness, served: `tensorfold lanes` with `--kernels native` (our glue, not a captured Triton set), 6 prompts of 5 to 5,000 tokens, 160 tokens each. Drafted, `--no-drafts` and `--solo` give the same token hashes on `0efbd447` and on this branch: 18 of 18 streams equal.
- Mutations: a norm divided by width + 1 fails `norms` and `norm-rows`. A 4-bit row reading the low nibble twice fails `norm-rows` and `glue`.
- Speed: no change served. Together, drafted: 4.744 s vs 4.752 s. No drafts: 10.719 s vs 10.752 s. Solo: 7.002 s vs 6.973 s (`0efbd447` first, then this branch). The embed itself is about 3× faster per call in an in-model profile (median 5.98 → 1.90 µs), but there are only 46 calls a run.
- Tests run: `zig build test` and `zig build test-cuda-host` (44/44 steps, 619 passed, 5 skipped as on `main`). `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: Metal (unchanged); the bf16 embed path and ModelOpt Nemotron (left as they are).

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (Served hashes equal across all three modes and both builds.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (Nemotron's bits do not change.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
