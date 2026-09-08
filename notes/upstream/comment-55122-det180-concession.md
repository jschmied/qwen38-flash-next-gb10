DRAFT — needs the user's go. vllm-project/vllm PR #55122, follow-up to LopezCastroRoberto's review.
Slot marked ⟨VPP7⟩ for the true-stock run still in flight; post either with that filled in, or with
that paragraph deleted if the run is inconclusive.

---

A correction to my own evidence, which runs against this PR rather than for it.

I have been measuring end-to-end reproducibility on GB10 with an arm I labelled "stock". It was not
stock. Our serving venv carries four determinism fixes and only two are env-gated, so disabling the
deterministic top-k left the other three active — bit-stable MoE finalize, a FlashInfer autotune
cache-key fix, and a PLE offload semaphore reset. I labelled that arm from the one switch I had
flipped rather than from what was actually installed.

Re-reading the results with the right label, eight sequential cases — 1,460 to 5,960 tokens, prefix
caching on and off, MTP on and off, three prompt shapes chosen to force ties at the selection
boundary — show **zero position-level divergence with the deterministic top-k disabled**, measured
with per-position `prompt_logprobs` across 8 repeats. So on this build, with those other three fixes
in place, **the top-k fix alone was not required for sequential reproducibility on real prompts.**

That is evidence for your position, not mine, and you should have it.

What it does not overturn: the kernel itself is still 0/81 self-consistent on synthetic tie-heavy
inputs, and @k3dani's 0/4 reproducible prompts were measured on an image predating all four fixes.
Both remain true. The reconciliation is the one #53287 already reached — the defect needs ties at the
top-k boundary, and real prompt score distributions may simply not produce them often enough to
matter. I could not manufacture a case that produced them end to end, across two prompt generators
and three tie shapes.

⟨VPP7⟩

Where that leaves this PR, from my side: the reproducibility argument for changing the **default** is
weaker than I presented it, and your opt-in shape is the better one. What I would still argue for is
that the deterministic kernel be *available* as the backend behind your flag on hardware where the
FlashInfer path is not — which on sm_121 is currently the case (`TopKRaggedTransform … operation not
supported`, reported above). The two perf commits stand on their own merits and I would keep them
regardless of which way the default goes.

<!-- AI disclosure: produced with AI assistance; I reviewed every line and ran every number quoted. -->
