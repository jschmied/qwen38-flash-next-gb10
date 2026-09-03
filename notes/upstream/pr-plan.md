# PR plan (only after the issue has a number, and only if a maintainer says which surface)

Files: `vllm/envs.py` (+1 env var, default `True`), `flashinfer_cutlass_moe.py` (+1 kwarg at the
`flashinfer_cutlass_fused_moe(...)` call). No kernel change. Title: `[Kernel] Allow disabling the
FlashInfer CUTLASS MoE fused finalize (deterministic output)`. Body: link the issue, the
before/after table, "includes AI-assisted code" note. Commit with `git commit -s`, trailer
`Co-authored-by: Claude`. Human runs the repro on both settings before pushing (policy: no pure
agent PRs; every changed line reviewed by the submitter).

Our local equivalent (env-gated on `VLLM_MOE_DET_FINALIZE=1`): `tools/determinism/fusedfinalize_patch.py`.

## Kernel PR (persistent_topk determinism)

Files: `csrc/libtorch_stable/persistent_topk.cuh`, `csrc/libtorch_stable/topk_histogram_4096.cuh`
(diffs in `patches/kernel-det/*.det.diff`). Add a CPU-free unit test in `tests/kernels/` that runs
the op 6× on random and tie-heavy inputs across rows {1, 8, 64} and lengths crossing 2048 / 8192 /
32768 and asserts bit-identity plus equality with an exact (value desc, index asc) reference —
`patches/kernel-det/test_det.py` is the template. Evidence: finding 53 (indexer first to differ with
identical inputs), 54 (exact selection makes the 7.5k forward bit-identical), the #54521 thread.
