POSTED 2026-09-27 (user's go: "ok"). vllm-project/vllm#53912, reply to ming616's cross-request report.

@ming616 One concrete mechanism that produces this symptom class, if your GLM-5.3 build runs RecoverSSM (compact KDA recovery; `--use-replayssm` upstream): in align mode, when a speculative window's accepted tokens end exactly on a recurrent block boundary, the commit plan and the postprocess kernel select the *next* block-table column for the committed state (`n // block` instead of `(n - 1) // block`):

- [`kimi_k3/nvidia/ops/recoverssm.py#L285-L288`](https://github.com/vllm-project/vllm/blob/24c9772d19251dbbf70fef75119546827c540c63/vllm/models/kimi_k3/nvidia/ops/recoverssm.py#L285-L288)
- [`v1/worker/gpu/model_states/recoverssm.py#L95-L99`](https://github.com/vllm-project/vllm/blob/24c9772d19251dbbf70fef75119546827c540c63/vllm/v1/worker/gpu/model_states/recoverssm.py#L95-L99)

That column can still be unallocated, so the commit is dropped while the running-state column is advanced to it. The next step reads whatever that page last held, which after a free can be another request's state, and a prefix-cache hit at that boundary reuses it. Nothing logs an error and acceptance looks normal; long contexts make it more likely because more boundaries are crossed.

Fix and test: [local-inference-lab/vllm@a02f585d](https://github.com/local-inference-lab/vllm/commit/a02f585d3f2418cb9f9e8d70c9229bb179b5f86c) (Martin Vit). The same change is in #58863 ([5567cc1b25](https://github.com/vllm-project/vllm/pull/58863/commits/5567cc1b25c6fe79b99d228068017e371efc2570)), where it also covers the Qwen GDN path; on our GB10 build (main 1ea7c63f4 + that PR) the fork's test fails 3 of its boundary cases without the fix and passes with it.

Does your fork pass `--use-replayssm` (or enable compact KDA recovery by default)? If not, this is not your path.

Written with AI assistance (Claude Code).
