POSTED 2026-09-30 (user's go: "do all important replies"). vllm-project/vllm#58835, reply to @antoniocuegervas.

@antoniocuegervas thanks for measuring it under load. All three waits land within 4 %, inside those cells' own spread, and the bound fires on 0.5 % of steps over five hours of agent sessions. Together with our one-stream result, that says the wait costs nothing measurable either way. It's also useful to know that most of the per-step wait is the worker waiting for the previous step's CUDA event, not the fill (6–32 ms). I'll rebase the PR, since it has conflicts with main again.

Written with AI assistance (Claude Code).
