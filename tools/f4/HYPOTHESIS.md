F4 (full CUDA graphs for the RecoverSSM verify path), 2026-09-28, before the run. User: "1. yes, do 4. next", then "all".
Builders gdn_recoverssm / ple_recoverssm report UNIFORM_BATCH (was NEVER, which forced PIECEWISE); padded graph rows
get null state slots and zero-length windows (kernel writes zeros and returns for state_idx <= NULL_BLOCK_ID).
Clone venv rssm, prod config (K=5, probabilistic drafts, bf16 SSM state, prefix-match-unit 64), KV 4 GiB, fastload on.
Arms: full (FN_CG_MODE=FULL_AND_PIECEWISE) vs pw (PIECEWISE), 2 starts each, alternating. Probe nvprobe.py.
H: greedy hashes identical between arms (same kernels, only launch changes); c=1 code/prose ms/tok -0 ... -4 %
(finding 237: FULL_DECODE_ONLY on the native path was null; c=1 duty cycle 54 % leaves launch gaps, but PIECEWISE
already graphs most of them); c=4 -0 ... -3 %; TTFT unchanged (+-2 %, prefill is PIECEWISE in both).
Out of range: any hash difference (padding/state bug), a slowdown > 2 %, or a gain > 6 % (then check what else moved).
Tests first: GDN/PLE/config RecoverSSM tests incl. the new padded-row test; a failure skips the A/B.
