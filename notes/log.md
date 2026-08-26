# Log

Newest first. Each entry records what was tried, what happened, and what it ruled out.

## 2026-08-26 — setup, and the source-build scare

- Model released ~12:29 UTC. `Qwen4ExpForConditionalGeneration`, no vLLM support at
  release; PR #53896 appeared the same day (111 files, unreviewed).
- Sized the builds. NVFP4 (`RadixArk`) at 125.9 GiB is the only one within reach of
  128 GB unified, and still ~9 GiB over what is usable.
- Found the split: main model 78.2 GiB, PLE/n-gram table 47.7 GiB in ten
  `model-plefp8-*` files. Main model alone fits with ~39 GiB to spare.
- Thought a source build was mandatory (`output_gate_type = sigmoid` vs a prebuilt
  kernel defaulting to `silu`). It is not — see
  [why-no-source-build](why-no-source-build.md). This was the decisive finding of the day.
- Prepared the venv (nightly `dev1244+g8d301f075`, chosen because it is 7 commits ahead
  of the PR base and 0 behind, so all 40 modified files match byte-for-byte).
- Staged the 73-file overlay: 0 empty, 0 syntax errors.
- Weights downloading (46% at time of writing).

**Not yet known:** whether the architecture loads at all on sm_121, and whether the
residency gap can be closed. Nothing has been booted.
