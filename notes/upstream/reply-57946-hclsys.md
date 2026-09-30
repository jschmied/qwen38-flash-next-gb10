POSTED 2026-09-30 (user's go: "do all important replies"). vllm-project/vllm#57946, reply to @hclsys.

@hclsys thanks, both are in `da6d4a0068`.

- **Per row, not per slot.** The comment and the PR description now say it: the router's mask is `is_padding[:num_tokens]`, so a marked row is `-1` in every slot and its whole output is discarded, and no live row loses a single expert.
- **The guard is skipped when `VLLM_MOE_SKIP_PADDING` is off.** Every writer of the sentinel is gated on that flag. b12x also refuses expert parallelism and expert maps (`supports_expert_map()` returns `False`), so no other path can hand it a `-1`. With the flag off, the guard is skipped by a static read resolved while the graph is traced.

`tests/kernels/moe/test_flashinfer_b12x_moe.py` on GB10 (sm_121): 51 passed with the flag on, 48 passed with `VLLM_MOE_SKIP_PADDING=0` (the three sentinel cases deselected).

Written with AI assistance (Claude Code).
