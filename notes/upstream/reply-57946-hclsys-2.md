POSTED 2026-09-30 as https://github.com/vllm-project/vllm/pull/57946#issuecomment-5911371374 — user's go 2026-09-30 ("push and short reply"). vllm-project/vllm#57946, second reply to @hclsys.

@hclsys right, my 48 came from `-k`, not from the file. `a2c8e66411` adds your `skipif`: whole file on GB10 now gives 51 passed with the flag on, 48 passed and 3 skipped with it off. The description now says an unguarded `-1` is an out-of-bounds access that kills the context. Thanks for running both legs.

Written with AI assistance (Claude Code).
