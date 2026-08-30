# The patch series, and where it can go

## There is no "main" to target

`vllm/models/qwen4_exp/` does not exist on vllm-project/vllm **main** — nor does anything
Flash-Next. Code search on the default branch returns zero for `qwen4_exp`,
`QSAMetadataBuilder`, `QWEN38NEXT_GEMM_PLANS`. No released version has it either (newest is
v0.28.0). The code lives only on PR #53896, which is open, conflicted, and 32 ahead / 49 behind
main.

Our serving venv is an unpacked wheel `0.1.dev20073+g8e685d198` taken from a vendor container;
that revision **does not exist in vllm-project/vllm** (HTTP 422). So every patch we hold is
written against a tree with no upstream identity, and the PR has since renamed the package
`qwen3_8_flash_next` → `qwen4_exp` and the dict `QWEN38NEXT_GEMM_PLANS` →
`QWEN4_EXP_GEMM_PLANS`.

## Two branches, two targets

**`fix-modelopt-pcpt-dispatch`** — off `upstream/main`, ONE commit, 2 insertions. The
`FP8_PER_CHANNEL_PER_TOKEN` dispatch gap **exists on main**, independently of Flash-Next:
`ModelOptFp8PcPtLinearMethod` is defined there and the algo string appears five times, but
`ModelOptMixedPrecisionConfig.get_quant_method` has no branch for it. This one is a normal
upstream bugfix PR against main — it does not need the Flash-Next PR to land. Exported to
`patches/series-main/`.

## Branch (Flash-Next specific)

`gb10-sm121-fixes`, rebased onto PR #53896 head **`91a6b555`** (was `82399a9`; the PR went CONFLICTING → MERGEABLE on 2026-08-29 and none of its new commits touch our four files), in `~/vllm-fork`. Three commits,
12 insertions. Exported to `patches/series-pr53896/` and verified to apply cleanly to a pristine
PR head with `git am --3way`.

| commit | what | evidence |
| --- | --- | --- |
| `[Quantization] Dispatch FP8_PER_CHANNEL_PER_TOKEN` | `ModelOptFp8PcPtLinearMethod` exists but is unreachable through `MIXED_PRECISION`; the algo resolves, matches no branch, falls through to `UnquantizedLinearMethod`, and the checkpoint then fails to load with *no module or parameter named ...weight_scale* | hit directly; FP8_PB_WO had the same omission and was fixed upstream since our wheel |
| `[Model] Pass quant_config through GatedResidual` | `quant_config=None` hardcoded on all three projections, so hyper-connections are always unquantized. The docstring claims the opposite | 25% of decode GPU time on GB10 sits in these layers' BF16 GEMMs, on `cutlass_80_wmma_tensorop_bf16` |
| `[Model] Pass quant_config to ParallelLMHead` | omitted at both sites; a quantized head loads unquantized | +11% single-stream on GB10, no measurable quality cost |

Deliberately **not** in the branch:

- **`_is_sm103()` widened to sm_121.** We measured it: 36.45 stock vs 35.92 with the kernel
  provably dispatching, n=6 each from empty compile caches — not significant. Proposing it would
  be proposing a change with no demonstrated benefit. Kept as
  `patches/skinny-gemm-sm121-gate.patch` with the four blockers written up in
  `skinny-gemm-on-sm121.md`.
- **QSA fused multi-step draft decode.** Correct but worth nothing (`patches/qsa-fused-draft-decode.patch`).

## Push is blocked, and not by us

```
! [remote rejected] gb10-sm121-fixes -> gb10-sm121-fixes
  (refusing to allow an OAuth App to create or update workflow
   `.github/workflows/pre-commit.yml` without `workflow` scope)
```

The branch carries `.github/workflows/` from the PR head; our token has
`admin:public_key, gist, read:org, repo` and no `workflow`. To push:

```
gh auth refresh -h github.com -s workflow     # interactive, needs a browser
cd ~/vllm-fork
git push -u origin fix-modelopt-pcpt-dispatch   # -> PR against main
git push -u origin gb10-sm121-fixes             # -> for the #53896 thread
```

Both branches hit the same rejection, and neither is our doing: our fork is behind
`upstream/main`, so any branch based on current upstream carries main's `.github/workflows/`
files as new, and the token has `admin:public_key, gist, read:org, repo` without `workflow`.

Until then the series in `patches/series-pr53896/` is the portable form — it applies to the PR
head with `git am --3way` and needs no fork access.

## Backported into our venv: `4e8b849b8d97` (PLE offload event pool)

Applied 2026-08-30 to `/opt/llm/runtime/vllm-venv-fnext`. **This is a venv-local edit and will be
silently reverted by any reinstall or upgrade of vLLM in that venv** — the same way the FP8 oracle
shim was lost on every upgrade. If a future start begins hanging at `warmup_kernels`, check this
first.

**What it fixes.** The shipped `0.1.dev20073+g8e685d198` tree allocates **one**
`_input_ready_event` and **one** `_d2h_done_event` per connector, on a stated assumption that only
one request is ever outstanding:

```python
# PLE rejects DBO, and each forward consumes its output before the
# next launch, so one pending request is sufficient.
self._request_queue = queue.Queue(maxsize=1)
```

Under async scheduling that is false — `max_concurrent_batches == 2`, the request thread pops
immediately so the queue is empty while request #1 is still staging, and `_launch(#2)` re-records
the *same* event behind forward #1's PLE semaphore wait. The offload thread then waits on a record
the model stream cannot reach until it sends request #1, while the main thread's device-wide
`torch.accelerator.synchronize()` at the end of `warmup_kernels` waits on the D2H stream. Deadlock,
with no transfer needing to be slow. Upstream replaces the shared events with a per-request pool
sized `max_concurrent_batches`, drops `_d2h_stream` and its cross-stream `wait_event`, and issues
the staging copies on the model stream.

**Why we applied it even though we never hung.** We are on the racy path — V2 Model Runner, async
scheduling on (it is the default; `"mtp"` is inside `EagleModelTypes` so speculation does *not*
disable it, and `mp` reports `supports_async_scheduling() == True`), so two batches in flight. Four
independent reporters on this code hang at startup and we do not. We were winning a race, not
avoiding one. See vllm#53960.

**Verification, in this order:**

1. The patched module is what actually imports — checked by `inspect.getsource`, not by trusting a
   `.pyc`: `_input_ready_event` absent, `_d2h_event_pool` present, `_PendingPleOffloadRequest`
   present.
2. It serves: ready in 680 s, PLE worker registers and reaches `Busy-loop started`.
3. It costs nothing: c=1 decode **38.1 / 36.5 / 34.8** against a 36.45 ± 1.04 reference
   (noise floor 6.9%).

Backups: `connector.py.pre-4e8b849`, `model_runner.py.pre-4e8b849` beside the originals.

⚠️ Applying it: `printf pw | sudo -S patch -p1 < file` **feeds the patch file to sudo as the
password** — the redirect beats the pipe. Use `patch -d DIR -p1 -i FILE`.

## Branch drift since our build (checked 2026-08-30)

#53896 head is now `11564b8869ea`. The commit itself is **CI-only** — two buildkite YAMLs and one
test — but checking it surfaced drift that matters more than the commit:

- **The package was renamed: `vllm/models/qwen3_8_flash_next/` → `vllm/models/qwen4_exp/`.**
  Every source path we cite is therefore build-specific. In particular the FP8-KV closure in
  `TODO.md` cites `vllm/models/qwen3_8_flash_next/nvidia/qsa.py:69-70`; on a current build that file
  is `vllm/models/qwen4_exp/nvidia/qsa.py`. The *finding* is unaffected — the guards are the same —
  but a reader on a newer build cannot find the file by the path we published.
- **`37c7110fa619` refactored the NGram helpers**, +108/−73 across `{amd,nvidia}/ple_layer.py`. That
  is in the branch but **not** in our `0.1.dev20073+g8e685d198` build, and our hand-applied
  `4e8b849b8d97` is against the older layout. Anyone upgrading should expect the backport not to
  apply cleanly, and should re-check whether it is still needed at all.
- **`vllm/v1/ple_offload/` does not exist on #53896's branch.** It belongs to #53899. Our venv is a
  preview build combining the two PRs, which is worth stating whenever we quote a path from it —
  neither PR alone reproduces our tree.
