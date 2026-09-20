# Upstream patches kept for possible later use

> **Audited 2026-09-20.** `vllm-pr50729-mamba-state-copy-race.diff` was **deleted**:
> [#50729](https://github.com/vllm-project/vllm/pull/50729) merged into `main` on 2026-08-17 and is
> in our dev524 build (`v1/worker/mamba_utils.py`). The other four stay:
>
> | file | upstream | why kept |
> |---|---|---|
> | `vllm-pr47861-eagle-peek-mamba.diff` | **closed unmerged** | a closed PR's diff may stop being fetchable; see below |
> | `vllm-pr48375-drop-eagle-block-mamba.diff` | [#48375](https://github.com/vllm-project/vllm/pull/48375) **open** | `disable_eagle_block_drop` is worth −26 % per warm turn here |
> | `vllm-pr53798-align-seed-mamba-block.diff` | [#53798](https://github.com/vllm-project/vllm/pull/53798) **open** | one of four competing fixes; see vllm#53142 |
> | `vllm-pr54076-align-split-mamba-block.diff` | [#54076](https://github.com/vllm-project/vllm/pull/54076) **open** | same family |


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

## `vllm-pr54076-align-split-mamba-block.dev524.diff` — rebased 2026-09-20, NOT yet applied

The upstream diff no longer applies to our serving build: hunk 1 wants to add a `MambaSpec` import
that `dev524` already has (line 59, for `prefill_checkpoint_alignment`), and hunk 4 predates
[#53614](https://github.com/vllm-project/vllm/pull/53614)'s internal-checkpoint exemption. Rebased
to 3 hunks / 48 lines: hunk 1 dropped, hunks 2-3 unchanged, hunk 4 reconciled to
`0 if use_internal_checkpoint else next_block_boundary` per wickist's 2026-09-06 note on the PR.
Verified by `ast` against the pristine file (56 methods, identical class structure) and by a clean
`patch --dry-run`. The derivation it adds (`mamba_state_block_sizes`) is genuinely absent from
`dev524`, so an unpatched arm really is unpatched.

**Precondition for measuring it — the default config is a no-op.** Our serve logs
`interface.py:933` "Setting attention block size to 1568 tokens" and `interface.py:957` "Padding
mamba page size by 0.13% to ensure that mamba page size and attention page size are **exactly
equal**". That is the condition wickist reported on 2026-09-09 from an RTX 3090 TP2 box: with the
grids equal the heterogeneous geometry never arises and the fix cannot do anything. Reaching the
regime the fix governs needs an explicit `--block-size` (their repro: 816 against a mamba spec of
1648). Measuring on the default config would produce a null that means nothing — the same shape as
the swizzle null recorded as finding 147.

**What is owed** (wickist, 2026-09-16): prefix-cache hit rate on an immediate same-prompt re-ask,
patched vs unpatched, EOS-correct harness, 3 starts, optionally a third arm with the scoping commit
(`use_eagle_preserves_target_kv_cache()`). Read it from `vllm:prefix_cache_hits_total` deltas —
`cached_tokens` is inert and returns 0 on provable hits, see [[prefix-cache-hit-measurement-trap]].
