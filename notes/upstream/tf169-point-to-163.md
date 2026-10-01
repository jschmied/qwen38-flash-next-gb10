DRAFT — needs the user's go. ashhart/TensorFold#169, short pointer to #163 (2026-10-01).

This looks covered by #163: its second half keeps prefill snapshots at message starts (the chat template's resume points), so a new conversation that forks after the shared system and tools block resumes from it, and under `--parallel` a fork takes a free lane. @ashhart said there that they'll rebase it into 0.6.1.

Written with AI assistance (Claude Code).
