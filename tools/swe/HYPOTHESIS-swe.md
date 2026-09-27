SWE (the deciding quality benchmark for the precision cuts), 2026-09-27, written before the run. User: "NVFP4 GDN need
reals becnhmark", "bf16 SSM state also need benchmark. we have to pick smallest quality trade".
Arms: fp8gdn (prod), nvfp4gdn, bf16ssm; exact prod config (K=5, dprob, RecoverSSM + boundary fix, block 1728) but
FN_MAXLEN 65536 and default KV (the smoke run hit ContextWindowExceeded at 32k with max_tokens 16000). 2 starts each.
Agent: mini-swe-agent 2.4.5 on the x86 box, fixed slices Verified-30 + Multilingual-28 (58 instances), -w 4,
reasoning_effort medium, 1.0/0.95/20, max_tokens 16000, step limit 250 (benchmark default).
Smoke (base, 2 instances): 183 s gen, 1 resolved, 1 context overflow (config, fixed).
H: resolve rates of the three arms within the run-to-run spread of prod (expected spread +-3 instances of 58 at
temperature 1.0); a cut is a NOTICEABLE loss only if both of its starts sit below both prod starts AND paired
per-instance losses outnumber wins clearly (sign test p<0.1 over the 116 paired instance-runs).
bf16ssm is the arm where long trajectories can show state-error accumulation; watch its empty-patch and
ContextWindow/format-error counts separately from resolve rate.
Out of range: any arm with > 2x prod's empty patches or error exits -> triage the trajectories before any verdict.
