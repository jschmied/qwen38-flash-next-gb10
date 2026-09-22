WITHDRAWN 2026-09-22 — posted, then DELETED by the user: it published a half-finished run
(no generation exercised) and explained my own harness race to upstream. Do not repost as-is;
re-run with the completion inside the runner first.


Ran it on `1ea7c63f4`. **The startup geometry persists on current main.**

```
Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.
Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block
Initializing a V1 LLM engine (v0.29.1rc1.dev533+g1ea7c63f4)
```

Same 800/200, same 1,228,800 bytes, same 37.50 % as your 0.28.0 excerpt and as my dev524 run. So
across three revisions — 0.28.0, `dev524` (2026-09-08), and main `1ea7c63f4` (today) — it is
unchanged.

**How it was run, since "on main" is the whole point.** The aarch64 nightly wheel, extracted and put
on `PYTHONPATH`, so the PLE-offload patches in my site-packages are shadowed — this model needs no
offload, so it runs on stock main. The runner logs `vllm.__version__` *and its path* before starting
anything, so a silent shadowing failure would show `0.28.1rc1.dev524` rather than quietly producing
a fake "main" result; it resolved to the wheel. Box: DGX Spark GB10, sm_121, TP=1, Qwen3.8-27B-FP8,
your config verbatim.

**Runtime correctness: still not tested, and your caveat stands.** The engine reached serving state
after 480 s — so workers and KV cache initialized, not merely config parsing — but I only checked
`/v1/models`. My runner tears the server down as soon as it is up, and my follow-up completion lost
the race with its own cleanup. If a generation check on main would be useful, say so and I will
re-run with the request inside the runner.
