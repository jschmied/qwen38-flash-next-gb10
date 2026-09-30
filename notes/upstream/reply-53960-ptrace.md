POSTED 2026-09-30 as https://github.com/vllm-project/vllm/issues/53960#issuecomment-5911536372 — user's go 2026-09-30 ("post"). vllm-project/vllm#53960, reply to @alansrobotlab2 / @estrella159 on ptrace_scope.

The `pidfd_getfd` requirement belongs to #53899's separate offload process, and #53899 closed unmerged on 09-21. Main's PLE offload (#54371, merged 09-09) runs in-process over UVA, so there is no second process to attach to, and on current main neither `ptrace_scope` nor `CAP_SYS_PTRACE` matters.

For anyone still on the #53899 stack, one more option keeps the host at scope 1: run the server as a systemd unit with `AmbientCapabilities=CAP_SYS_PTRACE` and `CapabilityBoundingSet=CAP_SYS_PTRACE`. The process gets that one capability and nothing else; our GB10 served that way from 09-20 until we moved to main.

On unified-memory boxes (DGX Spark), note that #54371's default pins the whole table, which does not fit 128 GB beside the weights; #58439 maps it from the checkpoint instead.

Written with AI assistance (Claude Code).
