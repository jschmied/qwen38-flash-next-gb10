# Reproducing this on your own DGX Spark

A working server, start to finish. The [README](README.md) says *what we measured*; this says
*what to type*. Most version pins below are load-bearing and have a documented failure mode. One of
them turned out not to be, and is marked as reversed rather than quietly deleted.

**Target:** `36.5 tok/s` single-stream, ~`100 tok/s` aggregate at 16 concurrent, 32K+ context,
tool calling, on one GB10 with 128 GB unified memory.

**Time:** ~5 h, almost all of it downloading. Budget ~3 h for the weights alone.

---

## 0. What you need

| | |
|---|---|
| hardware | NVIDIA DGX Spark (GB10, **sm_121**, 128 GB unified, aarch64) |
| disk | **~140 GB free.** The base is 123 GB and the overlay adds ~12 GB |
| **swap** | **64 GiB, and it is load-bearing — not a safety net.** Measured peak is 48.3 GiB. See below |
| vLLM | `0.28.1rc1.dev524+g5db652225` — **nightly `main` + our #53899 port** (see below) |
| FlashInfer | `0.6.18.post1` — python + cubin + jit-cache, all three at the same version |
| torch | `2.13.0+cu130` |

### Swap is part of the recipe on a 128 GB box

The PLE table is "offloaded to host memory", but on GB10 host and device are the **same physical
pool**, so offloading moves the allocation between accounting buckets without creating capacity.
The two loads also run **concurrently** — `PleOffloadWorker` streams its 10 PLE shards while the
main worker streams the other 196 — so the peak is additive:

| | |
|---|---|
| main model, device side | 78.2 GiB |
| PLE n-gram table, host side | 47.7 GiB |
| KV cache + CUDA context + runtime | ~20 GiB |
| **peak demand** | **~146 GiB** |
| `MemTotal` on a 128 GB Spark | **121.6 GiB** |

The difference has to be swap. Ubuntu's default `/swap.img` is 16 GiB and **is not enough**:
measured here, swap use passes 16 GiB while the main worker is still under 40 % loaded, and
plateaus at **48.3 GiB** — which is the PLE table almost exactly. The table is cold during load,
so the kernel pushes essentially all of it out as the main worker claims RAM, and `MemAvailable`
*recovers* from 7.9 GiB to ~38 GiB as it does. That is the mechanism working, not a warning sign.
Size swap to hold the whole table with headroom: 64 GiB.

With too little swap the PLE worker is OOM-killed at ~200/206 shards and the engine
reports only `RuntimeError: PLE offload worker exited during startup` (an `EOFError` on the
worker's pipe) — the kill is in `dmesg`, not in the vLLM log. That is
[vllm#53960](https://github.com/vllm-project/vllm/issues/53960).

```bash
# 48 GiB here + Ubuntu's default 16 GiB /swap.img = 64 GiB total
sudo fallocate -l 48G /swap2.img && sudo chmod 600 /swap2.img
sudo mkswap /swap2.img && sudo swapon /swap2.img
echo '/swap2.img none swap sw,pri=-2 0 0' | sudo tee -a /etc/fstab
```

Two consequences worth stating, because both cost us a box:

- **Do not disable swap.** Without it the load does not degrade, it dies.
- **Do not pin the PLE table.** Pinned pages are unswappable, so pinning defeats exactly the
  mechanism above and thrashes the box to the point of needing a hard reset. `main`'s own
  UVA-offload path pins; the #53899 worker keeps the table in ordinary anonymous memory, which is
  what makes it swappable.

### The build: `main` does not have PLE offload, and you have to port it

vLLM `main` has the model (#53896, merged 2026-08-31) but **not** PLE offload — #53899 is still
open and `vllm/v1/ple_offload/` does not exist upstream. Without it the 51.2 B-parameter n-gram
table stays resident on the GPU and the model does not fit. The generic UVA route
(`--cpu-offload-gb`) is not a substitute: it pins the tables and thrashes the box.

The two trees have diverged, so this is a real port, not a cherry-pick — but it is done and it
works. [`tools/main/`](tools/main/) carries it: apply `pr53899.vllm.diff` (13 files clean, 14/17
hunks of `ple_layer.py`), then `port53899.py` hand-ports the three rejects onto main's fused-op PLE
layer. [`tools/main/BUILD-RECIPE.md`](tools/main/BUILD-RECIPE.md) is the full procedure for bumping
to a newer nightly, with the traps that cost us time — **clone the serving venv, never build a
fresh one** (a fresh venv breaks the `torch 2.13.0+cu130` pin), and verify the clone by a log line
from the running server naming the venv path, because `cp -a` leaves the copy's interpreter
pointing at the original.

### Take FlashInfer 0.6.18, not 0.6.17

**This reverses what this file said until 2026-09-11.** We pinned 0.6.17 on the grounds that
flashinfer#4757 dropped SM121a from the aarch64 JIT-cache arch list, so 0.6.18 would lose the
prebuilt cubins and fall back to a ninja fan-out that once took this box down. We had read the PR
description and never opened the wheel. Opening both wheels refutes it:

| `flashinfer_jit_cache/` | 0.6.17 | 0.6.18.post1 |
|---|---|---|
| filenames containing `121` | **0** | **0** |
| `fp4_gemm_cutlass_sm120.so` | 17 × `sm_120` ELF | the same 17 × `sm_120` ELF |

Neither version ships an sm121 artifact — 0.6.17 predates #4757 and has none either, so there were
never any to lose. Nothing in the tree is gated to `sm_121a`: `compute_120f` covers CC 12.0 **and**
12.1, and `sm_121a` is required only for sparse MMA (`mma.sp .kind::mxf4nvf4`), which this model
does not use. Everything we touch is a `*_sm120` module, identical in both. Empirically, after
cutting over, `~/.cache/flashinfer/0.6.18.post1/121a/` holds **0 modules** and a full start log has
zero `ninja`/`nvcc`/`Compiling` lines. There is no JIT fallback here, so there is no OOM exposure.
(Measured in [det-208](notes/determinism-investigation.md).)

The pin was also expensive: vllm#55715 below asks for FlashInfer ≥ 0.6.18, so anyone following the
old advice read themselves out of a working kernel.

> **Startup takes ~12 minutes and always did.** The FlashInfer autotune cache is version-scoped
> (`flashinfer_autotune_cache/<version>/121a/<hash>/`), so an upgrade does discard the previous
> tuning — but it is not what you wait for. Measured startup to `Application startup complete`:
> **11:20 / 12:01 on 0.6.17**, **11:54 on 0.6.18.post1**. Loading 123 GB dominates; the version
> change is invisible next to it. Don't mistake the wait for a hang, and don't attribute it to
> the upgrade. Keep all three packages at one version:
> `flashinfer-python` from PyPI, `flashinfer-cubin` from `https://flashinfer.ai/whl/` (PyPI tops
> out at 0.6.13), `flashinfer-jit-cache` from `https://flashinfer.ai/whl/cu130/`. `jit/env.py`
> asserts cubin == python and aborts the import on a mismatch.

---

## 1. Weights

Start from `RadixArk/Qwen3.8-Flash-Next-NVFP4` (125.9 GiB). It splits into a main model (78.2 GiB,
196 files) and the PLE n-gram table (47.7 GiB, 10 files, already FP8) — the split is what makes the
model fit, because the table is offloaded to host memory.

```bash
# ~3 h. Cap the rate; the Xet CDN behaves badly with many connections.
aria2c -x1 -s1 --max-overall-download-limit=6M -i urls.txt
```

> ⚠️ **Verify `lfs.sha256`, never file size.** `aria2` preallocates, so a file reaches its final
> size the instant it starts. Two size-correct, byte-corrupt shards here produced *fluent garbage*
> that was invariant to every configuration change — it cost a full day and two retracted upstream
> issues. See [fetching a slice](notes/fetching-a-slice.md).

## 2. The derived checkpoint

The published NVFP4 checkpoint leaves the dense projections in BF16. Two changes take it from
17.1 to 26.1 tok/s at no measurable quality cost:

1. **dense projections → FP8** (from `lovedheart`'s mixed checkpoint) — same size, **+39%**
2. **`lm_head` → FP8** (ours) — **+11%**, and it *doubles* to +19.1% once MTP is on

Build it as a **hardlink overlay**, not a copy: only ~12 GB of tensors actually change, so
hardlink the other 411 files against the base and you spend 12 GB instead of 123.

```bash
scripts/quant_lmhead.py --base <base> --out <overlay>   # see notes/quantizing-lm-head.md
```

> ⚠️ **A hardlinked `config.json` is shared, and editing it in place edits the base too.** This
> silently corrupted our production checkpoint once: `config.json` showed `links=2`. Break the link
> before editing — `cp config.json config.json.new && mv config.json.new config.json`.

> ⚠️ **The runtime reads `config.json`'s embedded `quantization_config`, not `hf_quant_config.json`.**
> Editing the latter looks correct and changes nothing. And under `quant_algo: MIXED_PRECISION` the
> authoritative field is **`quantized_layers`**, not `config_groups` — writing the wrong one gives
> you a W4A4 kernel with an uninitialised activation scale: random logits, immediate end token,
> **zero characters of output, no error**. Gate every build offline before starting a server
> ([choosing a quant scheme](notes/choosing-a-quant-scheme.md)).

## 3. Patches

The serving venv carries **15 modified files** on top of the stock `dev524` wheel, installed by
[`tools/main/prod-det-overlays.sh`](tools/main/prod-det-overlays.sh). Eight of the fifteen are the
[#53899](https://github.com/vllm-project/vllm/pull/53899) PLE-offload port, which is **still
unmerged** — so there is no version of this recipe that runs on stock vLLM.

```bash
./tools/main/prod-det-overlays.sh        # idempotent; reports applied / already / FAILED per item
```

**Any `pip install`/upgrade of vLLM in this venv silently reverts all fifteen.** The symptoms are
non-obvious — a startup hang at `warmup_kernels`, HTTP 400 on every tool call, missing scale
parameters. Re-run the installer after any reinstall.

> **The older `patches/*.patch` + `apply.sh` bundle is gone** (archived 2026-09-20). It was cut
> against `0.1.dev20073` and every file in it targeted `vllm/models/qwen3_8_flash_next/...`, a path
> `dev524` does not have — so it could not be applied to the build this page describes. Four of its
> ideas are still live and are carried by `tools/main/` instead: both `lm_head` `quant_config`
> patches, the `GatedResidual` threading (currently **not** applied), and the QSA ring widening
> ([#54912](https://github.com/vllm-project/vllm/pull/54912), open).
> [`patches/MANIFEST.md`](patches/MANIFEST.md) remains as the record of what each change was for,
> with per-item upstream status.

To verify what is actually applied, diff the serving venv against the pristine wheel snapshot rather
than reading the patch files — that is how the `GatedResidual` gap above was found:

```bash
tar xzf /opt/llm/runtime/vllm-venv-fnmain3-vllm-pkg-pristine-dev524.tgz -C /tmp/pristine
LC_ALL=C diff -rq --exclude=__pycache__ --exclude='*.pyc' --exclude='*.orig*' --exclude='*.pre-*' \
  /tmp/pristine/vllm "$SP/vllm" | grep '^Files'      # LC_ALL=C: a localised diff breaks the grep
```

### One more thing to check: the GDN prefill kernel

Three of every four layers in this model are linear-attention (GDN). Until vllm#55715
(merged 2026-09-08, `f6326f53b`) `_resolve_gdn_prefill_backend()` set `supports_flashinfer` for SM90
and the SM10x family only, so **sm_121 silently fell through to the Triton/FLA fallback for every one
of them** — 1,216 of 1,216 logged backend announcements on this box, never once FlashInfer. It is a
silent default, not an error.

If your build is at or past `f6326f53b` you get it for free. Below that it is +10/−2 in one file;
ours is [`tools/main/`](tools/main/). Either way, **verify from the log rather than from the version** —
both the main worker and the PLE offload worker must say:

```
Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)
```

Measured here (det-207), 3 starts per arm, same venv both arms, only the backport differing —
ranges, not means, and the two arms do not overlap:

| | warm reps (prefix cache hot) | cold rep 0 (full 6,001-token prefill) |
|---|---|---|
| FlashInfer | **0.558–0.564 s** | **2.437–2.482 s** |
| Triton/FLA | 0.587–0.594 s | 2.610–2.687 s |
| | **+5.0 %** | **+7.1 %** |

Cold is the bigger win because on warm reps the prefix cache absorbs most of the prompt, so less GDN
prefill actually runs. That cold cell is the one comparable to upstream's **7.2 % at ISL 32768**, and
it lands on it. We did not measure 32k, and this kernel does not touch decode.

## 4. Serve

[`scripts/serve-flashnext.sh`](scripts/serve-flashnext.sh) is the live launcher from our box.

**Three source overlays are part of the recipe** (installed on the serving venv by
[`tools/main/prod-det-overlays.sh`](tools/main/prod-det-overlays.sh); any vLLM reinstall reverts them silently, so re-run it after
every upgrade): the deterministic `persistent_topk` and the bit-stable MoE finalize above (env-gated, the launcher defaults them
on), and the
**PLE offload semaphore reset** (`plefix_patch.py`, [PR #13 on the #53899 branch](https://github.com/peakcrosser7/vllm/pull/13)).
The last one is not optional: with CUDA graphs enabled the unpatched #53899 branch consumes the *previous* step's PLE outputs
on every forward — identical consecutive requests hide it, real traffic never does (finding 138 in
[`notes/determinism-investigation.md`](notes/determinism-investigation.md)).
The flags that are not obvious:

| setting | why |
|---|---|
| `VLLM_PLE_CPU_OFFLOAD=1` | the whole point; keeps the 51.2 B n-gram table off the GPU |
| `VLLM_USE_DEEP_GEMM=0` | DeepGEMM gates on device-capability *family* 120, which sm_121 satisfies — then faults with `unspecified launch failure` (vllm#54125) |
| `VLLM_GDN_DECODE_KERNEL=triton` | the default CUDA kernel deterministically hangs the engine at c≈32 with FP8 GDN projections. No error, requests just stall |
| `CUTE_DSL_ARCH=sm_121a` | required for the FlashInfer CuteDSL path |
| `VLLM_QSA_DET_TOPK=1` + `VLLM_QSA_DET_LIB=<path>/_C_det.so` | deterministic `persistent_topk` (index-ranked ties; `patches/kernel-det`, v2.4). Without it greedy prefill above the 2,048-token indexer budget is not reproducible (vllm#54521, fix upstream in vllm#55122) |
| `VLLM_MOE_DET_FINALIZE=1` | FlashInfer cutlass MoE with `use_fused_finalize=False`: bit-stable finalize (+3.6 % decode). The autotune cache key must include `use_fused_finalize` or the server dies with `Invalid gemm2 profile id` (vllm#54945). **FlashInfer ≥ 0.6.18 does this upstream** (`fused_moe/core.py`, flashinfer#3367); below that you need the backport — a fourth overlay we used to carry and have now retired (det-209) |
| `--max-model-len 32768` | **not 8192.** A single code task emitted 31,115 characters of *thinking* before 12,931 of content. 8192 cannot hold this model's own reasoning |
| `--max-num-seqs 16` | **not 2.** Our early "concurrency ceiling" was this flag, not the hardware — the box reaches 266.8 tok/s at 48 streams |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | without these, every request carrying `tools` returns **HTTP 400** |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | +35% — **under re-measurement**, this figure predates three fixes now in the serving path and was not taken at three starts. `k=5` hard-fails (QSA ring capacity must divide the attention block size); `k=0..4` and `9..12` are legal, `5..8` are not — a hole, not a ceiling |

> **Correction (2026-09-12).** This slot used to carry a warning never to combine MTP with
> `--async-scheduling`. **That warning was wrong and is withdrawn.** It was wrong twice over:
> `_prepare_ngram_context` is the PLE n-gram *embedding* context, not ngram speculation, and it reads
> a device tensor, not a CPU mirror. And it was never enforced in the first place —
> `SchedulerConfig.async_scheduling` defaults to `None`, the `None` branch *enables* it for MTP on a
> multiproc executor, and we never passed `--no-async-scheduling`. So **every MTP number on this page
> was measured with async scheduling ON.** Nothing here needs changing to reproduce it; the warning
> was the error. Pass `--no-async-scheduling` only if you want to A/B it.

**PLE offload needs ptrace permission, in Docker *and* on bare metal.** `rebuild_cuda_tensor` grabs
the GPU worker's fd with `pidfd_getfd`, and `PleOffloadWorker` is a **sibling** of that worker, not a
descendant of it — so the grab is a cross-process ptrace operation and is gated:

| | |
|---|---|
| Docker | `--cap-add=SYS_PTRACE` |
| bare metal | `sudo sysctl -w kernel.yama.ptrace_scope=0` (Ubuntu ships `1` in `/etc/sysctl.d/10-ptrace.conf`, and `1` means *descendants only*) |

Without it the engine loads all 206 shards, spends ~8 minutes doing so, and only then dies with
`RuntimeError: pidfd_getfd: Operation not permitted` wrapped in `PLE offload worker failed during
startup` — and, one frame up, the much less helpful `Failed core proc(s): {}`. Set it **before**
starting, or you pay the full load time to find out.

## 5. Verify — capabilities first, then speed

**A serving config has capabilities, not just throughput**, and no speed test sees them. Both of
the following were broken here for days while every benchmark looked fine:

```bash
# 1. tool calling -- must be 200, not 400
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{"model":"flashnext","messages":[{"role":"user","content":"hi"}],
      "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}'

# 2. a generation long enough to clear the thinking block
#    ignore_eos + assert completion_tokens > 0 -- max_tokens is a CEILING, not a target
```

> ⚠️ **An all-empty cell is not a comparison.** Twice here a determinism check reported five
> outputs "identical" when every one was the empty string — the model was still inside `<think>`
> and the budget ran out. Assert that you compared something: print character counts and refuse
> the verdict when the cell is empty.

Then speed. Expected, on the `fp8head` checkpoint:

| | |
|---|---|
| single-stream, MTP k=2 | **36.5 tok/s** |
| c=16, 4000-token inputs | ~100 tok/s aggregate |
| prefill | flat 2003–2380 tok/s from 4k to 60k context |
| decode | **depth-independent** — 26.8 → 27.1 across 15× context (the QSA signature) |

> **Noise floor is 6.9% for *decode*** (six identical runs: 34.7–37.1). Nothing under ~10% is
> callable from one run. **Prefill is far noisier: ±20%** — three runs of *one* config at 8k input
> spanned 1,633–2,367 tok/s. Treating the decode figure as general produced a withdrawn finding of
> ours on 2026-08-31.

---

## Things that look like levers and are not

Each measured null here, with the mechanism understood. Full detail in
[TODO](notes/TODO.md) under "closed".

- **Hyper-connection quantization or kernels** — 27% of decode time, three interventions, all null.
  They are latency-bound at ~78% of roofline: a quarter of decode because there are ~102,000 of
  them, not because any one is expensive.
- **NVFP4 KV cache** — two independent GB10 measurements plus a structural MTP-acceptance penalty.
  Fails silently.
- **Lowering `gpu-memory-utilization`** to avoid host freezes — refuted. `0.70` is the *worst*
  recorded outcome; the cause is absolute free memory at launch, not the ratio.
- **A "SM121 QSA kernel-guard fix" that widens the sm100→sm120 gate by family.** It is circulating
  in at least one popular DGX Spark repo. It was **retracted upstream** in sglang#36806: it routes
  to a path that corrupts output at long context — 1/4 runs wrong at 120K tokens, 4/4 at 210K,
  HTTP 200 throughout. Do not adopt it.

## Known-broken here, independent of anything you do

- **Greedy decoding is not reproducible on this model.** Same prompt, `temperature=0`, different
  output — from as few as 582 prompt tokens. On sm_121 `use_cooperative_topk` is False (the
  capability-*family* check excludes all of 12.x), so every request calls
  `torch.ops._C.persistent_topk`, which vllm#51782 reports silently returns wrong values in a
  data-dependent way.

  **Refinement (2026-09-12):** *calls* is not the same as *runs*. Inside `persistent_topk`, a row
  whose length exceeds `RADIX_THRESHOLD` needs a cooperative launch; if that launch would
  oversubscribe the device **and** `sharedMemPerBlockOptin < 128 KiB`, the C++ diverts to
  `top_k_per_row_decode` instead. GB10 reports **100 KiB**, so it meets that third condition
  always — meaning long rows here are served by a *different kernel* than short ones. That split is
  invisible at the Python level and it matters: vllm#55314, the fix for the bin-overflow defect,
  changes `persistent_topk.cuh` / `cooperative_topk.cuh` / `topk_histogram_4096.cuh` but **not**
  `sampler.cu`, where `top_k_per_row_decode` actually lives. Four other explanations were tested and eliminated
  ([write-up](notes/temp0-nondeterminism.md)).
- **Long-prompt hangs (>8k)** are `is_arch_support_pdl()` returning True for anything with
  `major >= 9`, so PDL is used in `_build_qsa_metadata_kernel` where the dependent kernel waits
  forever (vllm#53960).
