# Filtered-path benchmark for vllm#55122, on rented H100 / A100

## Why this exists

`persistent_topk` dispatches to `FilteredTopKRaggedTransform` when
`num_rows > 32 && sharedMemPerBlockOptin >= 128 KiB` (`topk.cu:38`). PR #55122 replaced that path's
per-row work with the rescanning `det_select_row`. **GB10 reports 99 KiB and never executes the
branch**, so every number in the PR comes from the persistent path and this one is untested by us —
it is the first item under "Hardware risks a reviewer should weigh" in the PR body.

Only A100 (163 KiB), H100/H200 (227 KiB) and Blackwell qualify. Ada (L4, L40S, RTX 4090), consumer
Ampere (A10, RTX 3090), RTX 5090, V100 and T4 all cap at 100 KiB or below and will report
`NOT reachable`.

## What it measures

Two arms, both built from source on the rented box, no vLLM install:

- `_C_det` — this PR (`persistent_topk.cuh` + `topk_det.cu` at branch head).
- `_C_base` — upstream at the merge-base `d9105ea8` (`base/`).

`run.py` prints the device's opt-in shared memory and refuses to be read as valid if the Filtered
path is not reachable, then runs correctness (rows 48/64, random / tie-heavy / all-equal) and cost
over the grid named in the PR body (64 rows x {4k, 8k, 20k, 40k} x k {512, 2048}), plus a sweep
across the `rows > 32` switch where the Filtered path turns on.

## Run it

    pip install modal
    modal token set --token-id <id> --token-secret <secret>   # headless; make the token in the web UI
    modal run modal_app.py                       # H100
    modal run modal_app.py --gpu A100-80GB       # A100

Output is printed and saved to `results/`.

## Or on any box you already have

    DET_ARCH=90a python build.py        # 80 for A100, 100a for B200
    python -c "import torch; \
      torch.ops.load_library('build/_C_det/_C_det.so'); \
      torch.ops.load_library('build/_C_base/_C_base.so'); \
      exec(open('run.py').read())"

## One edit to the baseline arm, disclosed

`base/topk_base.cu` is upstream's `topk.cu` with `#include "ops.h"` dropped, the exported symbol
renamed, and the `max_smem_per_block < 128 * 1024` fallback replaced by a hard error instead of the
call to `top_k_per_row_decode`. That branch is unreachable on the parts this bundle targets;
building it would drag in the whole decode kernel and `ops.h`. If it ever fires, the run fails
loudly rather than silently measuring a different path.

Both arms cross-compile clean for `sm_90a` and `sm_80` (verified on the GB10 with nvcc, compile
only — no Hopper hardware needed to check the build).

## Syntax-checking a branch without touching the GB10

`modal_app.py::check` compiles `topk_det.cu` for a given arch on Modal's CPU tier — `nvcc` needs no
GPU to compile, so this costs no GPU seconds and, the point, no time on the GB10 while it is
benchmarking. A compile error found here does not burn a queued A/B slot hours later.

    modal run modal_app.py::check --arch 121a

It compiles **whatever kernel this bundle holds**, and the checked-in `persistent_topk.cuh` is the
PR head that backs the H100/A100 numbers in `results/` — copy the branch's kernel in first if you
want to check something else, and revert it afterwards so the bundle stays reproducible.

Note (2026-09-08): the Modal token was revoked, so this path is unavailable until a new one is set.
