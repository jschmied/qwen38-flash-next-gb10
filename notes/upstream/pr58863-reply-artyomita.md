POSTED 2026-10-01 (user: "yes, only important posts"). vllm-project/vllm #58863 reply to ArtyomITA.

@ArtyomITA thanks, that was the right fix. Both changes are in [`514102a1db`](https://github.com/vllm-project/vllm/pull/58863/commits/514102a1db): the verify step records `exp(g)`, and the commit replays forward with `h = fma(c, k, h * decay)`. On a GB10 the FP32 kernel tests now require `torch.equal` against the native per-token states. The closed form failed 28 of the 62 cases; this passes all of them, including align boundaries and extreme gates. I didn't repeat your server check, and the description cites it.

Written with AI assistance (Claude Code).
