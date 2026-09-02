# PR body for #54552 — branch `jschmied/vllm:fix/qsa-ring-widen` (commit 02001b44)

Title: `[Bugfix] Widen the QSA raw-key ring instead of asserting divisibility`

Open with (after reviewing the diff yourself — vLLM policy):

```
gh pr create -R vllm-project/vllm --head jschmied:fix/qsa-ring-widen \
  --title "[Bugfix] Widen the QSA raw-key ring instead of asserting divisibility" \
  --body-file notes/upstream/pr-54552-body.md   # strip this header first
```

---

## Purpose

Fixes #54552. `QSAKeyStateCache.get_kv_cache_spec` asserts that the raw-key ring capacity divides
the attention block size. The safety property in the comment above the assert is one-sided: the ring
must be **at least** `span` rows; wider is slack. With `compress_ratio = 4`, `num_speculative_tokens`
5..8 (and 13..16) need a 12-row (20-row) ring, and neither the power-of-two block sizes nor the
hybrid LCM sizes 848/1616 have the factor, so those depths fail at engine init:

```
QSA ring capacity 12 must divide the attention block size 848
```

The change factors the arithmetic into `qsa_ring_capacity()`, which returns the smallest whole-group
ring `>= span` that divides the block size, logs once when it widens, and raises a `ValueError` with
the numbers when no such size exists. Every previously legal depth keeps its ring size, so the change
is a no-op for existing configurations.

## Test Plan

- New unit tests in `tests/models/qwen4_exp/test_config.py`: for block sizes 848 and 1616 and
  `num_speculative_tokens` 0..16 the ring is `>= span`, a multiple of `compress_ratio`, divides the
  block size, and equals the old value wherever the old value was legal; plus the concrete cases
  `(4, 5, 848) -> 16`, `(4, 8, 1616) -> 16`, `(4, 5, 16) -> 16`, and `(4, 13, 16)` raises.
- End-to-end on a GB10 (DGX Spark), Qwen3.8-Flash-Next NVFP4, MTP `num_speculative_tokens=5`,
  attention block size 1616: previously a hard init failure; with this change the engine starts, logs
  `QSA ring widened from 12 to 16 rows ...`, serves, and passes a needle-in-a-haystack retrieval
  check 5/5 at ~20k context. The same change has been running on my box for two days of benchmark
  loops (MTP n=5, prefix caching on) without errors.

## Test Result

```
tests/models/qwen4_exp/test_config.py: 10 passed
ruff check / ruff format --check: clean
```

Not verified: behaviour under draft *rejection* specifically with a widened ring, beyond the fact that
a wider ring cannot overwrite earlier than the minimal one (the property the assert protects).
@bojiang3 offered to reproduce on a GB10 on `main`.

## Disclosure

Includes AI-assisted code (Claude Code); every line reviewed and the tests run by me.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
