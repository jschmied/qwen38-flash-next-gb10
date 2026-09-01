# Upstream patches kept for possible later use

## `vllm-pr47861-eagle-peek-mamba.diff` — NOT APPLIED, and analysis says it would be a no-op here

[vllm#47861](https://github.com/vllm-project/vllm/pull/47861) fixed MTP + prefix-caching
correctness on hybrid Mamba models (tool-call leakage, needle-recall failures, degenerate
generations on cache-hit paths, ~20% accuracy drops). It was **closed unmerged**; only its
scheduler half landed via #51113, which **is** in our build (`_mamba_block_aligned_split`,
`mamba_partial_cache_hit`).

Kept because a closed PR's diff may not stay fetchable, and because it matches our symptoms:
recurrent state that cannot be rewound is exactly the shape of the one-way acceptance collapse
we measure (`notes/determinism-investigation.md`).

### Its core change

```diff
-  drop_eagle_block = use_eagle and idx not in eagle_verified
+  can_eagle_peek = use_eagle and spec.supports_eagle_cache_peek
+  drop_eagle_block = can_eagle_peek and idx not in eagle_verified
```

plus a `supports_eagle_cache_peek` property per spec. **`supports_eagle_cache_peek` does not
exist in our build**, so on the face of it the fix is missing.

### Why it is probably still a no-op for us

1. `use_eagle` is per group (`use_eagle = i in self.eagle_group_ids`), but
   `kv_cache_coordinator.py:110-111` **conservatively flags ALL groups** when per-group detection
   comes back empty — so a mamba group can get `use_eagle=True`, exactly as the PR feared.
2. **`MambaManager.find_longest_cache_hit` accepts `drop_eagle_block` and never reads it**
   (`single_type_kv_cache_manager.py:1392+` asserts only on spec type, DCP, PCP). The flag reaches
   the mamba path and does nothing.
3. The eagle *margin* is already excluded for `MambaSpec` at `kv_cache_coordinator.py:828`
   (*"No margin for mamba: its finder never drops"*).

So the PR gates the flag at the caller; our build neutralises it at the callee. The residual
difference is `eagle_verified` bookkeeping, which only decides whether the flag is set on a later
iteration — and the flag is inert for mamba either way.

### If applying anyway

The diff is against a much older tree and will not apply cleanly; hand-apply the three
`supports_eagle_cache_peek` properties plus the two-line gate. Do it on an idle box — patching the
venv while arms run corrupts them — and measure with `tools/determinism/` rather than by eye.
The claim to test is whether acceptance still collapses one-way over a long run
(`degrade.py`), not whether the prefill probe changes.
