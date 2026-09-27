SWESMOKE, 2026-09-27, before the run. Purpose: settle the SWE-bench settings for the quality-cut comparison
(user: NVFP4 GDN and bf16 SSM state need real benchmarks; pick the smallest trade). Base arm only (prod config, 4 GiB KV,
32k context), 2 Verified instances, mini-swe-agent 2.4.5 on the x86 box, reasoning_effort medium, 1.0/0.95/20, max_tokens 16000.
H: both instances produce a non-empty patch (medium effort converges on Flash-Next); gen 10-40 min per instance at -w 4.
Out of range (empty patches, context overflow at 32k, > 60 min/instance) -> change effort/budget/context before the full run.
Full-run size follows from gen_s: python-30 + javajs-28 per arm, arms base / nvfp4gdn / bf16ssm (both only if needed).
