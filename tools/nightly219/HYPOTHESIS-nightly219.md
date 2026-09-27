NIGHTLY219 (user: "forward prod to nightly and check if patches apply"), written 2026-09-27 before the run.
Arms: main1ea7 (prod venv, 1ea7c63f4 + prod overlays) vs nightly219 (a9eafde59 wheel + the same overlays forwarded, FlashInfer 0.7.0,
cutlass-dsl 4.7.1, humming 0.1.16), exact prod config (K5, dprob, RecoverSSM, block 1728), 4 GiB KV, 2 starts each.
H: starts and passes every prod path check (FNNVFP4 head, mapped PLE, GDN/PLE RecoverSSM, FNDRAFTPROB, K=5, capture sizes).
Speed: c1 code/prose within +-3 % of main1ea7 (no targeted upstream perf change for GB10/qwen4_exp in the 266 commits);
TTFT 8k/30k within +-5 % (#57105 preallocates the QSA prefill workspace: no speed effect expected).
Output: greedy c1 hashes may differ (FlashInfer 0.7.0 / cutlass-dsl kernels -> reduction order); acceptance within +-0.1.
Out of range: failed start, a missing path line, accept drop > 0.1, speed outside +-3 % -> triage before any prod talk.
