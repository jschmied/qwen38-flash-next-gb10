# Bumping the serving venv to a newer vLLM main

Supersedes the ad-hoc `buildfnmain2.sh`. Written 2026-09-08 while going dev401 → dev524; every
numbered trap below cost real time at least once.

## The shape of it

1. **Clone the current serving venv** — never build a fresh one. A fresh venv breaks
   `torch 2.13.0+cu130` (memory `vllm-026-cutover`). Clone, then rewrite the interpreter paths.
2. Download the nightly wheel for the target commit.
3. `pip install --no-deps` it into the clone.
4. Back up the pristine `vllm` package as a tarball (rollback without re-downloading).
5. Apply our overlay; hand-port whatever rejects.
6. Re-apply the prod determinism overlays **against the new venv**.
7. Verify, then cut prod over by changing `FN_VENV` — leave the old venv in place.

## Traps, in the order they bite

1. **`cp -a` of a venv starts the ORIGINAL's interpreter.** The copy's `bin/python` symlinks and
   `pyvenv.cfg` still point at the source. `buildfnmain2.sh` rewrites them with `sed`, which is
   right, but nothing *verified* it. Proof is a log line from the running server naming the venv
   path — not the absence of an error (memory `venv-copy-shebang-trap`, which invalidated 8
   measurement arms and 2 upstream reports).
2. **`prod_det_overlays.sh` hardcodes `vllm-venv-fnmain2`.** Running it after building a *new* venv
   silently patches the old one: the four det overlays go to the wrong place, the new venv comes up
   stock, and nothing errors. Pass the venv explicitly. Same for any other runner with a pinned
   `V2=`.
3. **The wheel index moved.** `https://wheels.vllm.ai/<sha>/` is now a directory tree
   (`cu130/ vllm/ xpu/`); the filenames live under `<sha>/vllm/` and point back to the flat
   top-level path. Scraping the top page finds nothing and looks like "no wheel published".
4. **`patch --dry-run` prints "checking file", not "patching file".** A script that greps for the
   latter reports `0 files patched / 0 failed`, which reads like a clean apply and is in fact a
   dry-run that matched nothing. Always assert the file count is non-zero before believing a
   reject count.
5. **Dry-run before applying.** We knew from `bumpdry` that 2 hunks in
   `vllm/models/qwen4_exp/nvidia/ple_layer.py` would fail; applying blind leaves `.rej` files and a
   half-patched module that imports fine.
6. **Disk.** A venv clone is ~16 GB and the box sits at 95 % used. Check before, not after.
   `/opt/llm/runtime/vllm-venv-fnmain` is the launcher's **default** (`FN_VENV:-...fnmain`), so it
   is not free to delete even though it is a previous generation.
7. **Heavy compiles hang the box** (unified CPU+GPU memory pool): `llm-switch stop` and
   `BUILD_JOBS=4` before any source build (memory `gb10-source-build-oom`). Wheel installs are fine.
8. **A bump invalidates comparisons.** Everything measured on the old build (det-169/171/172/173
   here) is not comparable afterwards — especially across vllm#55272, which removed torch.compile
   for Qwen3.8-Flash-Next. Finalise or plan to re-measure.

## What the overlay carries

17 files. Markers to grep for after applying: `LMHEADQ-MTP`, `SCALEINV-MTP`, `LMHEADQ (jschmied)`,
`SCALEINV`, `PleOffloadLayer`. Five paths (`vllm/v1/ple_offload/*`, `ple_offload_layer.py`) are
files the overlay *adds*, so they are legitimately absent from upstream — do not read that as a
missing dependency.

## Verification gate before cutting prod over

- `import torch, vllm` prints the expected versions
- `Qwen4ExpForConditionalGeneration` in `ModelRegistry.get_supported_archs()`
- 0 `.rej` files, all 17 overlay files `py_compile` clean
- all five markers present
- the det overlays are installed **in the new venv** (check the path in the script's output)
- a server start whose log names the new venv path, plus one real request

## dev524 (2026-09-08): the overlay shrank from 17 files to 12

Regenerating the overlay against pristine dev524 (`fnmain-overlay-dev524.diff`, 12 files, re-applies
with **0 failed hunks**) turned up something worth knowing: **upstream now ships
`vllm/v1/ple_offload/` and `vllm/model_executor/layers/ple_offload_layer.py`.** Those five files were
*additions* in the dev401 overlay; on dev524 they exist upstream and our copies are byte-identical to
them, so they carry no diff at all. The PLE offload machinery has landed (#53899 and successors) and
we no longer maintain it locally.

Two consequences:

- The dev401 overlay's "MISSING in one of the trees" rows for those paths were correct **then** and
  are wrong **now** — do not read them as a missing dependency on dev524.
- `pip install --no-deps` overwrites those files with upstream's versions. Memory
  `fnext-venv-ple-backport` warns that a hand-applied patch (`4e8b849b8d97`) on
  `v1/ple_offload/connector.py` reverts silently on reinstall and then startup hangs. On dev524 the
  hang did **not** occur (det-177: PleOffloadWorker spawns, server serves, output byte-identical to
  the old build on two shapes), which is consistent with that fix being upstream now — but the
  `.pre-4e8b849` backup still sits in the venv, so confirm rather than assume before relying on it.

Generate the overlay with, from a directory holding `newtree1` (pristine) and `newtree2` (patched):

    diff -ruN -x '*.orig' -x '*.orig-*' -x '*.pre*' -x '__pycache__' -x '*.rej' \
             -x 'qsa_indexer.py' -x 'flashinfer_cutlass_moe.py' newtree1 newtree2

The last two exclusions matter: `prod_det_overlays.sh` patches those separately, and folding its
changes into the base overlay would apply them twice and make the det arms un-switchable.
Verify with `patch -p1 --dry-run -N` from a tree containing `vllm/` — **`-p1`, not `-p2`**, and
`--dry-run` prints "checking file", so assert that count is 12 before believing "0 failed".
