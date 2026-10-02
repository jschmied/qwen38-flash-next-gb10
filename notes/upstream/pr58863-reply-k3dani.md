POSTED on vllm #58863 (2026-10-02, user: "post 3"): https://github.com/vllm-project/vllm/pull/58863#issuecomment-5953690720

@k3dani thanks for the independent backport, and for reproducing the align race against `3388ba1`.

Both observations are real, and the description now says so:
- **Block growth.** The replay record makes the unified Mamba page about 6 % larger, and the attention block follows it
  (1,600 → 1,664 tokens). Prefix reuse counts whole blocks, so a repeat can reuse up to one block less. That is your
  9,698-token case; your 85,890-token repeat reusing the same 83,200 in both arms fits.
- **Decode at K=2.** The gain comes from dropping the per-draft state write-back, so it grows with the draft length.
  Our runs were MTP K=3 and K=5. At K=2 with PIECEWISE there is little write-back to drop, so +1.9…+4 % at c=1 is
  plausibly the record and commit cost. I haven't measured K=2 myself.

@lucifer1004 merged main (`c3e2df3cea`) and dropped the `mamba_cache_mode "all"` row (`372a3c0c16`). On GB10 the GDN and
PLE RecoverSSM kernel tests and the RecoverSSM config tests pass (90).

Written with AI assistance (Claude Code).
