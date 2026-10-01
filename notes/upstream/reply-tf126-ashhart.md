DRAFT — needs the user's go. ashhart/TensorFold#126, reply to @ashhart after the landing in 0.6.0 (2026-10-01).

@ashhart thanks for landing it, and both changes are better than what I had: the head on its stored bytes halves the read, and FP8G prompts drop the bf16 rounding of the scales I'd flagged. I'll move our setup to 0.6.0 and re-measure Flash Next on it.

Written with AI assistance (Claude Code).
