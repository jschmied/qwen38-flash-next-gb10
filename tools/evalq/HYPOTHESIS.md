EVALQ (decision 3: GDN projections FP8 prod vs NVFP4 W4A16), written 2026-09-27 before the runs. User: "ok, do a eval test and compare results".
Config: exact prod config (prod venv main1ea7 + PR RecoverSSM, K=5, probabilistic drafting, launcher serve-flashnext.sh), 4 GiB KV, 2 starts per arm, alternating.
Eval: thinking off, greedy, c=16. GSM8K test 1319 (scored in the probe), HumanEval 164 (scored offline, pass@1). TTFT 8k/30k with nonce.
Noise floor = the two starts of the same arm (c=16 greedy is not batch-invariant): discordant items fp8gdn#0 vs #1.
H (quality): NLL +0.40 % vs BF16 predicts a task delta below the noise: GSM8K |Δ| <= 1.0 pp, HumanEval |Δ| <= 3 pp (5 problems);
  between-arm discordance within ~1.5x of the within-arm discordance. Out of range (NVFP4 worse by > 1.5 pp GSM8K on both starts,
  or discordance clearly above the floor) = real degradation.
H (TTFT): Marlin W4A16 dequant in prefill: NVFP4 GDN TTFT +0 ... +8 % at 8k and 30k (GDN projections are a minor share of prefill FLOPs).

AMENDED 2026-09-27 before any evalq run (user: "bf16 SSM state also need benchmark. we have to pick smallest quality
trade since they might multiply"; quality rule = no NOTICEABLE loss). Arms now: fp8gdn (base), nvfp4gdn, bf16ssm
(--mamba-ssm-cache-dtype bfloat16, block 1728 in every arm so only the dtype differs), both (nvfp4gdn + bf16ssm). 2 starts each.
H (bf16ssm): per-token perturbation ~1/5 of the 4-bit step (speed-of-light 4q-4r) -> GSM8K/HumanEval within the noise
  floor; short prompts (<1k) cannot show state-error accumulation, so a pass here is NOT a pass for long context
  (SWE-bench slices + long-context probe follow).
H (both): effects add, not multiply, at this size: |Δ| <= |Δ nvfp4gdn| + |Δ bf16ssm| + noise.
Decision rule: a cut survives only if no benchmark shows a loss beyond the within-arm floor on both starts;
between surviving cuts, prefer the one with the smaller measured Δ per unit of speed.
