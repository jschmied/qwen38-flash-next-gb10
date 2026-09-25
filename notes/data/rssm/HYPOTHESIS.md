# GDN RecoverSSM, phase 1 (mamba_cache_mode none), written 2026-09-25 ~22:45, before the run
Kernel test: verify outputs bit-identical to native, committed states within 1.4e-7 rel. Server: both arms on the
cloned venv vllm-venv-rssm, --no-enable-prefix-caching, KV 4 GiB; rssm arm FN_GDN_RECOVERSSM=1.
- H-speed: warm c=1 per verify cycle -0.8 to -1.5 ms (-1.5 to -2.7 %) and c=4 larger (-2 to -5 %): GDN writes drop
  from 12 MiB to ~3 MiB per layer per step plus a 3 MiB read in the commit.
- H-correct: greedy divergence vs base no earlier than the reduction-order reference (median >= 25 tokens), no
  degenerate text; within-arm hashes deterministic.
- Outside: crash/wrong text -> protocol bug (conv window, commit indices); slower -> commit/launch overhead dominates.

# GDN RecoverSSM, phase 2 (mamba_cache_mode align + prefix caching), written 2026-09-26 ~00:20, before the run
Phase 1 (start 1): c=1 per cycle 55.5 -> 54.2 ms (-2.3 %), c=4 per cycle -5.1 %; divergence vs base median 26 tok,
|dlogprob| 0.026 (reduction-order reference 31 / 0.023); text sane.
- H-speed: same sign and size as phase 1 at c=1 (-1.5 to -3 % per cycle), c=4 -3 to -6 %; align adds a boundary-state
  write in the commit only when a block boundary is crossed, so no more than +0.2 ms.
- H-correct: divergence vs base-align median >= 20 tok, text sane; prefix-cache replay must be self-consistent: the
  comboprobe's second (cache-hit) pass reproduces the first pass's hashes within the arm, as it does in base-align.
- H-cache: agent-loop s/turn no worse than base-align (+5 % max).
- Outside: hash mismatch between passes -> wrong boundary state written (commit ALIGN_MODE indices); crash at the
  PLE short-conv layer -> ple_recoverssm compaction.
