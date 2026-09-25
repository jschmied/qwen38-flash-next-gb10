# FNGDNCS: streaming stores for the GDN spec-decode state snapshots (written 2026-09-25 ~19:05, before the run)
Speed-of-light 4o: the GDN update writes 4 fp32 state snapshots (12 MiB) per layer per step in 42 us, faster than DRAM;
they sit dirty in L2 and their write-back costs the next kernels ~40 us/layer (out_proj 70->100 us, mixer 31->47 us),
~1.5 ms/step. `.cs` stores drain them during the GDN kernel instead. Same DRAM bytes, less interference.
A/B base vs cs, KV 4 GiB, decprobe twice per start (warm second pass measured), 2 starts per arm.
- H: c=1 ms per verify cycle -0.3 to -1.2 ms (-0.5 to -2 %) in every start pairing; c=4 similar; output hashes
  identical (values unchanged).
- Outside: cs slower -> the GDN kernel itself becomes DRAM-bound and pays more than the aftershock saved; then the
  only lever is fewer bytes (ReplaySSM #49887 or fewer/narrower snapshots).

## Result: null (c=1 per cycle Δ -0.02 ms, hashes identical). Next: fp16 SSM state cache (written ~20:35, before the run)
`--mamba-ssm-cache-dtype float16` (model default fp32; kernel still computes in fp32): snapshots 12 -> 6 MiB/layer,
initial-state read 3 -> 1.5 MiB. Dose-response (4o) predicts ~ -20 us/layer ~ -0.7 ms/step.
- H-speed: c=1 per cycle -0.4 to -1.0 ms (-0.7 to -1.8 %), 2 starts per arm, sign holds in both pairs.
- Outputs will differ (numerics). If the speed gain holds, a quality check (logprob divergence vs fp32 states)
  decides; if the speed gain is < 0.3 ms, stop here and record that only ReplaySSM-style designs remain.

## Short-horizon result + long-horizon test (lpq, written ~22:25, before the run)
divq (8 prompts x 512 greedy tokens): fp16 first divergence median 24.5 tokens, |dlogprob| before it mean 0.037 / p99 0.56;
the reduction-order reference (FNBF16SK) 31 tokens, 0.023 / 0.42 -> short-horizon deviation comparable (~1.6x).
Long horizon: 3 documents (~7-10k tokens), teacher-forced prompt logprobs, FN_BATCH=256 so the GDN state is stored and
reloaded every 256 tokens (~30-40 fp16 round trips per doc), no prefix caching. Arms base, ssm16, sk.
- H-benign: fp16 mean |dlogprob| vs base is flat across position buckets (no growth from the first to the last
  quarter, <= 1.5x) and <= 2x the reference arm's; total NLL change within +-0.5 %.
- H-drift: fp16 |dlogprob| grows with position (last quarter >= 2x the first) or NLL rises > 1 % -> accumulated state
  error; fp16 not acceptable without a real eval.
