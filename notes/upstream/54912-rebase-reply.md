POSTED 2026-10-05 19:47 (https://github.com/vllm-project/vllm/pull/54912#issuecomment-5999973986). vllm-project/vllm#54912 reply to @stecasta.
Thanks, especially for checking the rejection path on a GB10 and for tracking down why the block-size alignment never takes effect.

- **Rebased** onto main. There are now 3 commits: the two original ones and the nit. The only conflict was the test's import, after main moved `Qwen4ExpConfig` to transformers. `qsa_cache.py` merged cleanly with #58961.
- **Nit:** the widening log now lives in `qsa_ring_capacity`. It still returns an int, so `get_kv_cache_spec` and the tests are unchanged, and the span/minimal arithmetic exists in one place only.
- **Tests:** `tests/models/qwen4_exp/test_config.py` passes 15/15, ring tests included. `test_qsa_reference.py` passes 73/74. The one failure is an environment issue on my side (`vllm.vllm_flash_attn.layers` not built in this checkout), and it fails the same way on unmodified main.

AI assistance (Claude Code) was used for this update.
