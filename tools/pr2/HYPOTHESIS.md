# jschmied/vllm#2 (lucifer1004's three fixes for #58863) on GB10 (2026-09-30, user: "interrupt work, do both now")

Clone venv (prod K5 config, F4 FULL graphs, KV 4 GiB), one start per arm: base (the PR head as installed) vs pr2
(the three commits applied with `patch` onto the overlaid venv files; `recoverssm_gdn.py` equals the fork branch).
- Greedy hashes identical (d102a738 / 38c70791 as every K5 run on this venv): the commit change is the same
  arithmetic in the same order, the guard and teardown touch no math.
- Speed: code c=1 null…−1 % (the commit is ~0.3 ms of a ~64 ms cycle on H200; smaller share here), c=4 ±3 %.
- Memory: if our PIECEWISE (mixed/prefill) captures keep the profiling KV cache alive as on H200, pr2's MemAvailable /
  CUDA free at ready is higher by roughly the profiling cache (≥ 1 GiB); null if our explicit KV size avoids it.
Tests after install: their teardown test, test_gdn_fused_mtp.py, test_recoverssm_gdn.py and test_recoverssm_config.py
(the F4 versions from branch pr58863-next).
