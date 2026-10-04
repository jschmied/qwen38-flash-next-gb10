POSTED 2026-10-04 17:28 (https://github.com/vllm-project/vllm/pull/58863#issuecomment-5981600919). vllm-project/vllm#58863 reply to @TSUMUGI-XE.
Thanks, this is useful. Here's what I did with each part:

1. **Pointer tables: adopted** in a77830311b, with you as co-author. I used a small helper, `recoverssm_ptr_table`, rather than switching to `uint64`. It keeps the tables `int64` but stores the uint64 bit pattern, the same way #48109 / `_reinterpret_u64_as_i64` handles the Mamba state pointers. That way the kernels and the stride tables are untouched, and nothing changes on CUDA. Both sites now use it: the GDN commit context and the PLE `self.base`. There's also a CPU unit test with top-bit pointers. The GDN + PLE RecoverSSM tests pass on GB10 (74 passed).
2. **XPU dispatch to the Triton/FLA core:** I'd rather keep this PR NVIDIA-tested, since I can't run XPU here. A separate XPU follow-up with your `forward_xpu` routing and your B70 numbers would be very welcome.
3. **PP > 1 (recorded step tied to the `PPHandler` slot):** also a good fit for a follow-up. Please do share it.

AI assistance (Claude Code) was used for this reply.
