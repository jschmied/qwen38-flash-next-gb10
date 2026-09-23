POSTED 2026-09-23 as vllm-project/vllm#58439. vllm-project/vllm PR from jschmied:pr/ple-checkpoint-mapped (2026-09-23), updated after two reviews.
Review commit: jschmied/vllm:pr/ple-checkpoint-mapped @ 12689c164 (one squashed commit on upstream main 711fc55c1), after review 3.
Earlier heads: 0886ea160 (squash of c929c09a9), c929c09a9 kept as pr/ple-checkpoint-mapped-history. Not opened upstream.

# [Qwen4Exp] Checkpoint-mapped PLE storage for unified-memory GPUs (DGX Spark)

## Purpose

Qwen3.8-Flash-Next cannot be served on a unified-memory GPU such as **DGX Spark (GB10, 128 GB shared by CPU
and GPU)** with either of the storage options #54371 introduced for its 47.7 GiB FP8 PLE table:

- **device-resident:** the table does not fit next to the ~72 GiB of weights;
- **pinned host (`cpu_offload`, the default):** a 47.7 GiB anonymous host copy of the table, in the same physical
  pool as the weights and the KV cache. It cannot be cheaply dropped and re-created under memory pressure. On a
  128 GB Spark, the pinned table plus weights plus KV has hard-reset the machine.

This PR adds a third, opt-in option, `--engram-config '{"checkpoint_mapped": true}'`, which **reads the table
in place from the checkpoint's safetensors files**. On GPUs that access pageable host memory through the host
page tables (`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`), the lookup kernel reads a
read-only mapping of the files. There is no table-sized device or pinned allocation, no resident duplicate of
the table, and no CPU-gather/H2D staging path; the kernel still copies the selected rows into its output
buffer. The table's pages are clean, file-backed page-cache pages that the kernel can drop and re-read, shared
by every process that maps the same files.

**Design:**

- **New backend.** `Qwen4ExpPLEPageableHostEmbedding`, a third backend in the #54371 hierarchy. It reuses the
  pinned backend's buffer, ETP reduction and finalize path; only the storage and the gather differ.
- **Gather.** A Triton kernel gathers rows byte-wise through a per-shard base-address table, because the
  safetensors data offsets are not aligned. It writes zeros for rows outside the rank's ETP range; the
  existing ETP all-reduce combines ranks.
- **Current stream.** The lookup runs on the current stream, not the pinned backend's side stream. With the
  inherited side-stream lookup, greedy outputs on GB10 were not reproducible within one server start
  (identical prompts, cold vs warm: 2/8 equal, |Δlogprob| up to 1.41). The observed cause on GB10: the side
  stream reads the ids tensor from the CUDA-graph memory pool after the graph has released it, and later
  segments overwrite it at replay. Reading a persistent copy instead made it reproducible (8/8); #57785's capture-time sync did not
  (1/8). On the current stream: 8/8, |Δlogprob| 0, and identical across fresh starts.
- **Host-side page prefetch.** GPU faults on non-resident file pages are serviced one page at a time. A cold
  30k-token prefill (≈480k rows) took **88 s** with GPU faults alone, and **1.6 s** when 64 CPU threads fault
  the step's rows in first. The prefetch is driven from `Qwen4ExpModelState.prepare_inputs`, from a CPU hash
  of the same inputs, and only when `checkpoint_mapped` is set. It is only a hint: the lookup is correct
  whether or not it has finished.
- **File resolution.** Files come from the default loader's preparation, so `--download-dir`, subfolders and
  the `model.safetensors.index.json` filter apply exactly as for loading. Every shard that owns rows must exist
  once, with its exact shape and dtype; nothing is truncated.
- **Formats.** FP8 and unquantized (BF16) tables. `--load-format dummy` maps private anonymous zero pages.
- **Reload.** On a reload every incoming PLE shard is compared **in full** with the newly mapped files.
  `reload_weights(weights_path=...)` streams those files, so it matches and remaps the new checkpoint (during
  reload processing, or at the latest on the next lookup). PLE weights delivered from memory (weight sync)
  differ from the files and are rejected. A rejected load stays rejected rather than falling back to the
  previous mapping, and binding commits its state only after the mapping succeeded. The cost is one extra
  pass over the table, on reloads only.
- **Guards.** Rejected with `dp_shared_memory` (mapped pages are already shared), with `embedding_across_dp`
  (the host prefetch only sees this DP rank's requests), and for architectures without this backend (DeepSeek
  V4.1). A warning is logged on non-integrated GPUs that report the attribute, such as Grace Hopper or Grace
  Blackwell: **validated on DGX Spark only.**

## Test Plan

**Unit tests** (`tests/models/qwen4_exp/test_ple_pageable.py`, wired into the existing `models_basic` Qwen4Exp
job):

- CPU tests, 19:
  - discovery across files, short last shard, other layers ignored;
  - refusals: missing, short-coverage, extra, short-interior and duplicate shards, dtype mismatch;
  - index-filtered discovery;
  - CPU views address the checkpoint rows;
  - the zero mapping commits no memory when read (RSS);
  - config rejections (DeepSeek V4.1, `dp_shared_memory`, `embedding_across_dp`);
  - reload A→B remaps to B; an in-memory reload is rejected, including one with a single changed row that
    no sample would hit; a rejected load stays rejected and a later reload from disk recovers.
- GPU tests, 5, skipped without pageable access (the prefetcher test without CUDA):
  - bit-exact FP8/BF16 gather with an ETP range over a 0xFF-poisoned allocation;
  - CUDA-graph replay switching valid ids to invalid ids yields zeros, not stale bytes;
  - the zero mapping;
  - short-then-long steps through the host prefetcher's pinned staging.
- Each regression test fails on the behaviour it replaced (checked by mutating the fixed line back).

**End-to-end on DGX Spark** (GB10, sm_121, TP=1, driver 580.178, CUDA 13.0, torch 2.13, this branch's code on
the `1ea7c63f4` nightly wheel). Qwen3.8-Flash-Next with an NVFP4 body, the FP8 PLE table (47.68 GiB,
320,001,536 × 160 B in 128 shards / 10 files), MTP n=3. `--kv-cache-memory-bytes` 31 GiB unless noted.

## Test Result

**Unit tests:**
- `models_basic` Qwen4Exp job set (`test_config.py`, `test_ple.py`, `test_ple_pageable.py`): 49 passed,
  34 skipped (GPU-only) on CPU.
- `test_ple_pageable.py` on GB10: 24/24 (19 CPU + 5 GPU) on this commit.
- pre-commit clean.

**Memory.** There is no working baseline on unmodified main (see Purpose), so the reference is the previous
offload implementation (#53899's worker, never merged) serving the same checkpoint:

| | #53899 worker (reference) | `checkpoint_mapped` |
|---|---|---|
| swap used, steady state | 50–53 GiB | 5–6 GiB |
| table-sized pinned allocation | none (anonymous table, 47.7 GiB) | none (4 small pinned staging slots) |
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
- Every recorded serving step of a mixed prefill/decode workload: 2,500 steps, 11.36 M rows, recorded with
  stream-ordered async copies and no sync before the consumer. The ids equal a CPU hash of the recorded inputs,
  and the rows the model consumed equal the checkpoint bytes, **including all 255 mixed prefill+decode steps**.
  The only id mismatches were 8 CUDA-graph capture-time calls, where the ids buffer is not yet computed; those
  are not serving steps.
- Greedy outputs are identical across fresh starts and cold vs warm (8/8, Δlogprob 0).
- A BF16-table copy of the checkpoint (rows = the FP8 path's `fp8 * scale` in BF16, mapped as 320 B rows)
  generates **bit-identical** output to the FP8 table: 8/8 sequential prompts and 8/8 determinism probes.

**Auto KV sizing** (no `--kv-cache-memory-bytes`): KV sized itself to about 33 GiB. 20 min of c=8 diverse
traffic (2k–60k-token prompts): 248 requests, 0 errors, MemAvailable ≥ 2.0 GiB, swap ≤ 6.8 GiB, memory PSI
`full avg10` ≤ 6.8 % (p90 3.3 %). The speed numbers above use an explicit 31 GiB KV cap, so both arms are
compared at equal KV.

**Not covered:**
- TP>1 on hardware (single-GPU box; the ETP range is unit-tested).
- Grace Hopper / Grace Blackwell.
- NVFP4 tables (main has no NVFP4 PLE method).
- Model Runner V1: the lookup is correct there, but the host prefetch hooks into the MRV2 model state, so cold
  prefill is slow.
- Kernel-format weight reload (a full table cannot be copied into the mapping; this fails loudly).

## Comparison with other PLE-table storage implementations

What each implementation documents, plus what we measured ourselves on GB10. Other PRs' numbers are not repeated
here, and nothing was measured for #54129.

| | where the table lives | who gathers rows | copies per step | extra process / privileges | target hardware | status |
|---|---|---|---|---|---|---|
| #54371 device | GPU memory | GPU kernel | — | — | any CUDA-alike | merged; on GB10 does not fit next to the weights |
| #54371 pinned host (default) | pinned host memory (full table) | GPU kernel via UVA, on a side stream | — | — | UVA-capable | merged; on GB10 the 47.7 GiB pinned table plus weights plus KV exceeds the unified pool |
| #53899 offload worker | anonymous host memory in a CPU worker process | CPU | rows over IPC to the GPU | 1 process, `CAP_SYS_PTRACE` | any | closed unmerged; **was our GB10 production path: 50–53 GiB swap** |
| #54070 disk dir | file mapping of a table written at first boot to `VLLM_PLE_DISK_OFFLOAD_DIR` | CPU (the #53899 worker) | as #53899 | as #53899 | any | open, but built on the closed #53899 |
| #54129 mmap | read-only mapping of the checkpoint safetensors (page cache) | CPU worker threads at input preparation | staged H2D copy into stable GPU buffers | — | any GPU (MRV2) | open |
| #57497 ROCm | pinned host memory | GPU via UVA | — | — | AMD | open |
| **this PR** | read-only mapping of the checkpoint safetensors (page cache) | **GPU kernel reading the mapping directly**; CPU threads only fault pages in | none beyond the kernel writing the selected rows | — | GPUs that read pageable memory through host page tables (validated: DGX Spark) | — |

**Relation to #54129.** Both keep the table file-backed and reclaimable in the page cache, and share the
discovery and validation concerns; this PR borrows #54129's approach to resolving and validating checkpoint
files.
- **#54129** is the portable path: a CPU gather plus a host-to-device copy, which works on discrete GPUs.
- **This PR** is the unified-memory variant: where the GPU can dereference pageable memory, the gather reads the
  mapping directly, with no host-side gather or staging copy.

The two could share discovery code, or `checkpoint_mapped` could become the unified-memory mode of one
file-backed backend. We are happy to rebase onto #54129 if it lands first. SGLang has an NVMe-backed path for
the same table (sgl-project/sglang#36567).

## Open design questions

1. **File resolution.** It reuses `DefaultModelLoader._prepare_weights`, a private method, to get exactly the
   files the loader read (download dir, index filter). Should the loader expose the resolved file list instead?
2. **Host page prefetch.** A 64-thread CPU prefetcher lives in `Qwen4ExpModelState`. It is only a hint; the
   lookup is correct without it, but cold prefill is about 50× slower. Is the model state the right home, or
   should it hang off a runner hook?
3. **Selection.** Opt-in today. Should it be auto-selected on integrated GPUs that report pageable host-page-table
   access, where the pinned default does not fit?
4. **Relation to #54129.** One file-backed backend with two gather modes, or two backends?

## Related

- **#54371:** the storage hierarchy this extends.
- **#54129 (mmap PLE table, open):** also file-backed, via a CPU gather plus H2D copy. This PR is the
  unified-memory variant: the GPU reads the mapping directly, so there is no host gather and no staging copy.
  The validation approach borrows from it.
- **#57785 (open):** eager-break side-stream work left unjoined at capture. Tested with the inherited
  side-stream lookup on GB10: it does not remove the non-reproducibility above, whose cause is graph-pool
  reuse of the ids at replay. The stock pinned backend runs the same start_prefetch/finalize flow. That could
  not be tested here (its pinned table does not fit), so no claim about it is made.
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

cc @peakcrosser7 (#54371) @Trosfy (#54129)

**AI assistance:** this PR was developed with Claude (Anthropic). The human submitter reviewed every changed
line, ran the tests and measurements above on the hardware named, and takes responsibility for the change.
