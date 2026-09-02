# PR plan (only after the issue has a number, and only if a maintainer says which surface)

Files: `vllm/envs.py` (+1 env var, default `True`), `flashinfer_cutlass_moe.py` (+1 kwarg at the
`flashinfer_cutlass_fused_moe(...)` call). No kernel change. Title: `[Kernel] Allow disabling the
FlashInfer CUTLASS MoE fused finalize (deterministic output)`. Body: link the issue, the
before/after table, "includes AI-assisted code" note. Commit with `git commit -s`, trailer
`Co-authored-by: Claude`. Human runs the repro on both settings before pushing (policy: no pure
agent PRs; every changed line reviewed by the submitter).

Our local equivalent (env-gated on `VLLM_MOE_DET_FINALIZE=1`): `tools/determinism/fusedfinalize_patch.py`.
