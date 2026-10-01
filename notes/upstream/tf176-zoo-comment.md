DRAFT — needs the user's go. ashhart/TensorFold#176, comment with real-weight evidence (2026-10-01).

Checked against real checkpoints on one DGX Spark: on 0.6.0, a ModelOpt export with W4A16 NVFP4 DeltaNet projections
and one with NVFP4 MTP dense layers both fail at load after most weights are in, with
`RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 1280 but got size 2560` from
`b16_from_rows` (the packed rows are K/2 bytes). On this branch both are refused at the check in seconds, naming the
first layer (`...layers.0.linear_attn.in_proj_qkv`, `model.mtp.fc_embedding`). Our other variants (block-FP8 or bf16
DeltaNet, bf16 or block-FP8 head, NVFP4 MTP experts, bf16 n-gram table, the EXL3 pack) load and reply as on main.

Written with AI assistance (Claude Code).
