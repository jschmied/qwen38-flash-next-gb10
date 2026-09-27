PMU (vllm#54458 follow-up: prompt-end GDN state via --prefix-match-unit 64), 2026-09-27, before the run. User: "issue 54458
sounds like worth to fix". Scoping memo (i54458/design-memo.md): main can already save the GDN state at the exact prompt
end when the prefix match unit (default = the 1728 block) is smaller than the Mamba block; with MTP + disable_eagle_block_drop
it needs #58368 (merged 09-25, in nightly a9eafde59, NOT in prod base 1ea7c63f4).
Arms (exact prod config, K=5, 4 GiB KV, 2 starts): nvdef = nightly219 default unit; nvu64 = nightly219 + --prefix-match-unit 64;
m1u64 = prod venv main1ea7 + unit 64 (counterfactual: without #58368 the tail state is registered one unit low -> no gain).
Probe turnreplay.py: (A) 20k cached prefix + N in {16,256,1024}; (B) 2 held-out SWE trajectories replayed as a growing prompt.
H (B, nvu64 vs nvdef): recomputed tokens/turn median ~1,000 -> 100-350 (-65...-90 %); TTFT median -20...-35 %
  (finding 142: recompute shrinks faster than TTFT; small-M prefill + one extra stop-at-prompt-end forward, est. 40-80 ms).
H (A): N=16 intercept falls by 150-350 ms (recompute of the ~1728-block tail disappears).
H (m1u64): recomputed and TTFT within noise of nvdef-like behaviour (no gain) -> proves #58368 is the mechanism.
Out of range: nvu64 no gain (check hits), or a gain on m1u64 (then #58368 is not the mechanism).
Correctness: final_greedy_sha may differ between arms (resume point changes chunking; numeric drift is allowed by the
quality rule) but must be identical across the two starts of each arm.
