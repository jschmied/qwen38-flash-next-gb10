DRAFT — needs the user's go. vllm-project/vllm PR from jschmied:pr/ple-checkpoint-mapped (2026-09-23). Items marked TBD are filled from the round-2 results before any go.

# [Qwen4Exp] Checkpoint-mapped PLE storage for unified-memory GPUs (DGX Spark)

## Purpose

Qwen3.8-Flash-Next cannot be served on a unified-memory GPU such as **DGX Spark (GB10, 128 GB shared by CPU
and GPU)** with either of the storage options #54371 introduced for its 47.7 GiB FP8 PLE table:

- **device-resident:** the table does not fit next to the ~72 GiB of weights;
- **pinned host (`cpu_offload`, the default):** the pinned copy lives in the same physical memory. It can be
  neither swapped nor reclaimed, so it competes directly with the weights and the KV cache. On a 128 GB Spark
  this has hard-reset the machine.

This PR adds a third, opt-in option, `--engram-config '{"checkpoint_mapped": true}'`, which **reads the table
in place from the checkpoint's safetensors files**. On GPUs that access pageable host memory through the host
page tables (`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`), the lookup kernel reads a
read-only mapping of the files. The table then uses no device memory, no pinned memory and no copy. Its rows
are clean, reclaimable page-cache pages, shared by every process that maps the same files.

**Design:**

- **New backend.** `Qwen4ExpPLEPageableHostEmbedding`, a third backend in the #54371 hierarchy. It reuses the
  pinned backend's buffer, ETP reduction and finalize path; only the storage and the gather differ.
- **Gather.** A Triton kernel gathers rows byte-wise through a per-shard base-address table, because the
  safetensors data offsets are not aligned. It writes zeros for rows outside the rank's ETP range; the
  existing ETP all-reduce combines ranks.
- **Current stream.** The lookup runs on the current stream, not the pinned backend's side stream. With the
  side-stream flow, greedy outputs on GB10 were not reproducible within one server start (identical prompts,
  cold vs warm: 2/8 equal, |Δlogprob| up to 1.41). On the current stream: 8/8, |Δlogprob| 0, and identical
  across fresh starts. #57785 describes a side-stream hazard in breakable graph capture that fits this
  (see *Related*).
- **Host-side page prefetch.** GPU faults on non-resident file pages are serviced one page at a time. A cold
  30k-token prefill (≈480k rows) took **88 s** with GPU faults alone, and **1.6 s** when 64 CPU threads fault
  the step's rows in first. The prefetch is driven from `Qwen4ExpModelState.prepare_inputs`, from a CPU hash
  of the same inputs, and only when `checkpoint_mapped` is set. It is only a hint: the lookup is correct
  whether or not it has finished.
- **File resolution.** Files come from the default loader's preparation, so `--download-dir`, subfolders and
  the `model.safetensors.index.json` filter apply exactly as for loading. Every shard that owns rows must exist
  once, with its exact shape and dtype; nothing is truncated.
- **Formats.** FP8 and unquantized (BF16) tables. `--load-format dummy` maps private anonymous zero pages.
- **Guards.** Rejected with `dp_shared_memory` (mapped pages are already shared) and for architectures
  without this backend (DeepSeek V4.1). A warning is logged on non-integrated GPUs that report the attribute,
  such as Grace Hopper or Grace Blackwell: **validated on DGX Spark only.**

## Test Plan

**Unit tests** (`tests/models/qwen4_exp/test_ple_pageable.py`, wired into the existing `models_basic` Qwen4Exp
job):

- CPU tests, 14:
  - discovery across files, short last shard, other layers ignored;
  - refusals: missing, short-coverage, extra, short-interior and duplicate shards, dtype mismatch;
  - index-filtered discovery;
  - CPU views address the checkpoint rows;
  - the zero mapping commits no memory when read (RSS);
  - config rejections.
- GPU tests, 4, skipped unless the GPU reports pageable access:
  - bit-exact FP8/BF16 gather with an ETP range over a 0xFF-poisoned allocation;
  - CUDA-graph replay switching valid ids to invalid ids yields zeros, not stale bytes;
  - the zero mapping.

**End-to-end on DGX Spark** (GB10, sm_121, TP=1, driver 580.178, CUDA 13.0, torch 2.13, this branch's code on
the `1ea7c63f4` nightly wheel). Qwen3.8-Flash-Next with an NVFP4 body, the FP8 PLE table (47.68 GiB,
320,001,536 × 160 B in 128 shards / 10 files), MTP n=3. `--kv-cache-memory-bytes` 31 GiB unless noted.

## Test Result

**Unit tests:**
- `models_basic` Qwen4Exp job set (`test_config.py`, `test_ple.py`, `test_ple_pageable.py`): 44 passed,
  33 skipped (GPU-only) on CPU.
- GPU tests on GB10: 18/18.
- pre-commit clean.

**Memory.** There is no working baseline on unmodified main (see Purpose), so the reference is the previous
offload implementation (#53899's worker, never merged) serving the same checkpoint:

| | #53899 worker (reference) | `checkpoint_mapped` |
|---|---|---|
| swap used, steady state | 50–53 GiB | 5–6 GiB |
| pinned memory | 0 | 0 |
| extra processes | 1 (offload worker) | 0 |

**Speed**, two fresh starts per arm:

| | reference | `checkpoint_mapped` |
|---|---|---|
| TTFT, 3 prompts of 24–29k tokens, sum | 32.99 / 33.47 s | 31.77 / 31.97 s |
| TTFT, 3 prompts of 7–8k tokens, sum | 9.27 / 9.37 s | 9.01 / 9.02 s |
| decode, c=16, tok/s | 197.6 / 201.1 | 199.1 / 198.7 |

**Correctness:**
- Rows handed to the model equal the checkpoint bytes: in-server check on 6 steps per start, including
  4,096-token prefill chunks.
- Every recorded step of a mixed prefill/decode workload: ids equal a CPU hash of the recorded inputs, and the
  rows the model consumed equal the checkpoint bytes. **TBD (round-2 I1).**
- Greedy outputs are identical across fresh starts and cold vs warm (8/8, Δlogprob 0).
- A BF16-table copy of the checkpoint (rows = the FP8 path's `fp8 * scale` in BF16) generates identical output
  to the FP8 table. **TBD (round-2 I3b).**

**Auto KV sizing** (no `--kv-cache-memory-bytes`), 20 min of c=8 diverse traffic: KV 33.38 GiB. MemAvailable
min, swap max and PSI **TBD (round-2 I3c).**

**Not covered:**
- TP>1 on hardware (single-GPU box; the ETP range is unit-tested).
- Grace Hopper / Grace Blackwell.
- NVFP4 tables (main has no NVFP4 PLE method).
- Model Runner V1: the lookup is correct there, but the host prefetch hooks into the MRV2 model state, so cold
  prefill is slow.
- Weight reload / weight transfer.

## Related

- **#54371:** the storage hierarchy this extends.
- **#54129 (mmap PLE table, open):** also file-backed, via a CPU gather plus H2D copy. This PR is the
  unified-memory variant: the GPU reads the mapping directly, so there is no host gather and no staging copy.
  The validation approach borrows from it.
- **#57785 (open):** eager-break side-stream work left unjoined at capture. **TBD:** whether it removes the
  side-stream non-reproducibility on GB10 (round-2 I2).
- **#58310, #56926:** host-memory guards and serialization for the pinned path. They are orthogonal.

---

<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR.
- [x] The test plan.
- [x] The test results, before/after.
- [x] Documentation update: `docs/features/engram.md`.
- [ ] (Optional) Release notes update.
</details>

**AI assistance:** this PR was developed with Claude (Anthropic). The human submitter reviewed every changed
line, ran the tests and measurements above on the hardware named, and takes responsibility for the change.
