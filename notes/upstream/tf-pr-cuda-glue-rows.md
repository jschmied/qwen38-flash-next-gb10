# TF PR draft: cuda act_rows + gdn_prep + qk_rows (shared, operator-only) - NOT posted

Branch cuda-glue-rows (worktree ~/git/tf-glue), commit 28e8a6a1 on origin/main c36b536d (1.0.9). Not pushed yet.

Title: cuda: SwiGLU, output gates, Gated DeltaNet inputs, QK norm and RoPE for every family, checked against float64

## What this changes

Next to `norms` and `embed_rows` (#736), the per-row kernels a hybrid attention/DeltaNet family needs between its
projections, as shared modules, operator-only (no model calls them yet), in the shape of #746:

- `zig/kernels/cuda/act_rows.cu`, `zig/src/cuda/act_rows.zig`: SwiGLU from `[gate | up]` rows, the sigmoid output
  gate (`o * sigmoid(g)` from `[q | gate]` heads), 64-value group sums of the stored bf16 outputs (for an FP8/NVFP4
  activation scale downstream), and DFlash2's two-tap dynamic conv over block segments, with an optional residual.
- `zig/kernels/cuda/gdn_prep.cu`, `zig/src/cuda/gdn_prep.zig`: Gated DeltaNet inputs per row: the causal conv window
  over the kept rows plus the new ones (up to 8 taps), SiLU, q/k L2 norms, v, `g = exp(-exp(A_log) softplus(a))` and
  `beta = sigmoid(b)`; key dims 64, 128, 256.
- `zig/kernels/cuda/qk_rows.cu`, `zig/src/cuda/qk_rows.zig`: QK RMS norm (fp32 or bf16 weights), RoPE (rotate-half,
  full or partial, from a cos/sin table or from `inv_freq` in fp32), and the K/V cache write, from packed or strided
  sources (`[q | gate]` rows, padded k), packed or head-major outputs.
- `zig/src/core/gdn_prep_ref.zig`, `zig/src/core/qk_ref.zig`, and `swiglu`/`dconv` in `core/act_ref.zig`: the float64
  references.

Every kernel sums a row in one fixed order, so a row's bits do not depend on how many rows run beside it.

Users we know of: the Qwen3.8-27B family in #608 (its target's MLP, attention output gate, GDN layers and q/k, and
its DFlash2 drafter's conv), and any CUDA family with GDN layers or gated attention.

## Receipt

- Environment: TensorFold `c36b536d` (1.0.9) plus this commit; one DGX Spark (GB10, sm_121, CUDA 13). No checkpoint.
- Exactness, each at 1, 3, 17 and 64 rows with the bits of every smaller row count equal to the 64-row call's:
  - `tf-cuda-test act-rows`: SwiGLU at 17,408 and 192, output gate 24x256 and 3x64, group sums, dconv (5,120 in
    groups of 16, segments 16 and 7, with and without residual): worst 0.996 of the tolerance.
  - `tf-cuda-test gdn-prep`: 16/48 heads at 128 (the 27B), 2/3 at 64, 1/2 at 256: worst 0.995.
  - `tf-cuda-test qk-rows`: 48/4x128 full rotation, 48/4x128 none, 8/2x64 partial, 16/2x128 without norm, 24/4x256
    rotary 64 (the 27B), strided sources and head-major outputs bit-equal to the packed call: worst 0.988. The worst
    elements are the float64 value correctly rounded to bf16 just above a power of two; every element above 0.8 of
    the tolerance equals `bf16_rn(want)`.
- Mutations, each failing its test: the sigmoid's exponent scaled by 1.0625 (38x the tolerance), DFlash2's conv
  reading across a segment start (1,804x), softplus switching to `x` above 2 instead of 20 (14,503x), the QK norm
  dividing by `dim - 1` (1.43x).
- In a model (fork only, not in this PR): #608's 27B engine on these kernels with #746's attention, no Triton set:
  serial replies equal 0.6.6's except two at bf16 near-ties, drafted == serial, `bench_concurrent --alone --serial`
  12/12 cells exact; teacher-forced on 4,092 rows: NLL +0.0011 against the Triton set, argmax agreement 99.12 %,
  KL 0.00071. On an RTX 5090 (sm_120) every `tf-cuda-test` of these modules passed, mutations included
  (BobClawblaw, #608).
- Speed in this PR: none claimed, nothing calls it.
- Tests run: `zig build test` and `zig build test-cuda-host` (27/27 steps, 323/323 tests). `zig fmt --check` clean.
  `lean_check.py`: 538 findings, as on `main`.
- Not run, and why: Metal (unchanged); no model on `main` uses it.

`norm_rows_tests.zig` exports its test rig (`pub`) so `act_rows_tests.zig` shares it instead of copying it.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (No
      serving path changes.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why.
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
