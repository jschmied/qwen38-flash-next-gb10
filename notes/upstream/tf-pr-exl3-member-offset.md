POSTED 2026-10-01 (user: "191 is merged, do a folow up"). ashhart/TensorFold PR #193, base pr-141-0.6.1, head jschmied:exl3-member-offset 5727ea6.

Follow-up to #191, from its review: stacked on #191, the new commit is the last one.

#191 removes the grouping's shared-memory ceiling, so windows past it become reachable. `grouped_kernel` still looked
its members up with a 32-bit offset, `members[u * maxm + m]`. 2,049 experts at 1,048,576 rows overflow it at
u = 2,048, which the current checks accept. The offset is now `size_t`, as the fill kernel's writes already were. The
other offsets on these paths (rotation, both epilogues, combine, the activation rows) were already 64-bit.

Flash Next's 1,024- and 2,048-row windows were never affected.

On a GB10: test_exl3_experts, test_qwen4_exp_exl3 and test_group_kernel_smem 77 passed, 52 skipped. `routed()` hashes
are unchanged at 1, 8, 64 and 1,024 rows. There's no test at the boundary, since the members buffer alone would be
about 8.6 GB.

Written with AI assistance (Claude Code); the author reviewed every change.
