SYNCPROBE (2026-09-28 ~17:45; decides the early-exit design, §5v addendum). One host sync per verify cycle, placed
where an early-exit design would read the draft count: right after speculator.propose() in sample_tokens
(env FN_SYNCPROBE=1: torch.cuda.current_stream().synchronize(); logs once "FNSYNCPROBE sync after propose").
Clone venv rssm with prod's config (K=5, probabilistic drafts, bf16 state, pmu 64, FN_HCFUSE=1,
FULL_AND_PIECEWISE), KV 4 GiB, 2 starts per arm, probe nvprobe.
H: +0.5..+1.5 ms per verify cycle at c=1 (§5b's per-step syncs cost ~0.9 ms each): code +0.8..+2.3 % ms/tok (cycle
~64 ms), prose +0.7..+2.2 %; c=4 +0..+2 %; outputs identical (a sync cannot change numerics).
Reading: <= 0.9 ms -> the one-sync early exit is worth +2.7 % code / +5.9 % prose (replay); >= 2 ms -> +0.9 / +3.9 %,
then only the padded-row design is left.

VOID run 1 (syncprobe): the arms inherited the PIECEWISE arm's forbidden "Capturing CUDA graphs (FULL)" while also
requiring it (my spec error). syncprobe2: forbidden entry removed, every arm checked for required/forbidden overlap.
