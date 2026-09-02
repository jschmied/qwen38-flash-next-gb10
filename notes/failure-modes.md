# Failure modes: Qwen3.8-Flash-Next on a single GB10

> **Repo deleted 2026-08-28.** `Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark` now 404s. Its figures (34.8 tok/s free-form, 157.1 aggregate at c=8) and its HashK PLE compression are cited below and in other notes; they are no longer verifiable at source. Treat them as recorded-but-unverifiable rather than deleting them — they informed real decisions here.

Everything that went wrong getting this model to serve on one DGX Spark, organised by **what you
observe** — because that is how you arrive here. Each entry gives the signature, the cause, the
fix, and how to tell it apart from the ones that look identical.

Sources are marked: **[us]** hit and fixed here, **[field]** reported by another single-Spark
project and reproduced or credited, **[method]** a mistake in how we investigated rather than in
the software.

> **The single most useful line in this file:** when a symptom is **invariant to every
> configuration you change**, stop varying configuration. Invariance is evidence for corrupt
> *data*, not for an environmental cause. See A1.

---

## A. It loads and serves, but the output is wrong

Four distinct causes with near-identical presentation. Distinguish them before you start
eliminating hypotheses, or you will do what we did and eliminate twenty of them against the wrong
one.

| # | Output looks like | Cause |
|---|---|---|
| A1 | **fluent** sentences, wrong content, invariant to config | corrupt weight shard |
| A2 | **fluent** garbage, only under PLE CPU offload | shadowed `weight_scale` |
| A2b | **fluent** garbage from a well-benchmarked checkpoint | `FP8_PB_WO` layers loaded as BF16 |
| A3 | first token correct, then `!!!!` forever | SM121 kernel emitting NaN |
| A4 | correct text, but `</think>` leaks into `content` | no reasoning parser |

### A1. Corrupt shard — fluent garbage, invariant to everything **[us]**

**Signature.** The model loads without error. Tensor shapes and dtypes are sane. Activations have
correct magnitude. Output is *fluent* — grammatical, well-formed token salad, not NaNs and not
repetition loops. Critically, **it does not change** when you vary cudagraph mode, executor
backend, attention backend, quantization backend, batch size, prefix caching, eager mode, or
driver version.

**Cause.** One or more weight files downloaded to the correct *size* with the wrong *content*.
In our case 2 of 206: `model-bf16-00011.safetensors` (dense BF16 body) and
`model-plefp8-00000.safetensors` (PLE shards 0-12) — the two still in flight when the download
stalled, whose `.aria2` control files we then deleted, destroying the resume state that would
have caught it.

**Fix.** Verify against HuggingFace's published `lfs.sha256`, which is in the same API response
you are probably already parsing for sizes:

```bash
curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true&blobs=true" \
  | jq -r '.[]|select(.lfs)|"\(.lfs.sha256)  \(.path)"' > SHA256SUMS
sha256sum -c SHA256SUMS
```

**Why it is so hard to see.**

- Size checks pass. aria2 **preallocates**, so a file reaches its final size the instant it
  starts — size agreement proves only that bytes were reserved.
- The invariance reads exactly like an environmental or kernel fault, which sends you hunting
  through backends and driver versions. Every clean elimination makes the wrong conclusion look
  *better* supported.
- **It survives naive content checks.** We validated the PLE against the official BF16 table and
  got cosine 0.999635 — sampling row 0, which sat in the intact head of the corrupt file. Sample
  at several offsets, including near the end, or checksum instead.
- Corruption in **two different components** defeats the obvious bisection. Our body-vs-PLE split
  was the right experiment and could not separate them, because both halves were damaged.

### A2. Shadowed `weight_scale` under PLE CPU offload **[us]**

**Signature.** Fluent garbage, but *only* with `VLLM_PLE_CPU_OFFLOAD=1`. No error, no warning.

**Cause.** If the GPU-side process registers `weight`/`weight_scale` parameters for the PLE
embedding, they are never filled — `load_weights()` retains only `_offload_weight_scale`, because
the offload worker owns the real weights. The registered-but-unloaded `weight_scale` then
**shadows** `_offload_weight_scale` in `_get_embedding_weight_scale()`, and the lookup is
dequantized against an uninitialised value.

This is a trap you can walk into *while fixing B1*: our first version of that patch introduced
exactly this bug.

**Fix.** In `_get_ple_embedding_quant_method()`, return `None` in the GPU process when offload is
active, so nothing is registered:

```python
if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
    return None
```

### A2b. Blockwise-FP8 layers loaded as BF16 — the `FP8_PB_WO` dispatch gap **[us]**

**Signature.** Fluent garbage, whole-checkpoint, from a checkpoint whose published quality metrics
are excellent. Server starts clean; nothing in the log complains.

**Cause.** ModelOpt `MIXED_PRECISION` checkpoints may declare per-layer
`quant_algo: "FP8_PB_WO"` — 2D blockwise (128x128) weight-only FP8.
`ModelOptMixedPrecisionConfig.get_quant_method()` dispatches `FP8`, `NVFP4`, `W4A16_NVFP4` and
`MXFP8`, then **falls through**:

```python
if quant_algo == "MXFP8":
    return ModelOptMxFp8LinearMethod(self.mxfp8_config)
# Layer not in quantized_layers -- leave unquantized
return UnquantizedLinearMethod()      # <- FP8_PB_WO lands here
```

`UnquantizedLinearMethod` then loads the **packed FP8 bytes straight into BF16 parameters**. Every
value is reinterpreted garbage, and because it is a weight-loading path rather than a kernel
error, nothing raises.

**Present in vLLM `main`, absent from older snapshots.** Our tree
(`0.1.dev20073+g8e685d198`) already contains `ModelOptFp8PbWoLinearMethod` — the dispatch branch
is simply missing from the mixed-precision path. One line restores it:

```python
if quant_algo == "FP8_PB_WO":
    return ModelOptFp8PbWoLinearMethod(self.fp8_config)
```

**Check before serving any mixed-precision checkpoint**, offline, in seconds:

```bash
python - <<'EOF'
import re, vllm.model_executor.layers.quantization.modelopt as m
s = open(m.__file__).read()
i = s.index("class ModelOptMixedPrecisionConfig"); j = s.index("def get_quant_method", i)
print(re.findall(r'quant_algo == "([A-Z0-9_]+)"', s[j:j+2600]))
EOF
# the list must contain every quant_algo your checkpoint's hf_quant_config.json uses
```

Then cross-check against the checkpoint:

```bash
jq -r '.quantization.quantized_layers[].quant_algo' hf_quant_config.json | sort -u
```

The same gap exists in stock SGLang, where `FP8_PB_WO` likewise falls through to an unquantized
method — documented by the checkpoint author, which is how we learned to look.

`ModelOptFp8PbWoLinearMethod` also requires **both** dimensions divisible by 128. Verify against
the checkpoint's own shapes before committing to a download.

**The general rule this belongs to:** a quantization format that a runtime does not recognise is
far more dangerous than one it rejects. Rejection is an error message; non-recognition is a
silent reinterpretation of the bytes. Always enumerate what your runtime dispatches and intersect
it with what your checkpoint declares — this is a ten-second offline check that prevents a
day-long hunt.

### A3. flashinfer trtllm-gen decode kernels are SM100-only **[field]**

**Signature.** The first token is correct, then `!!!!` forever (NaN collapses to token 0).

**Cause.** trtllm-gen decode kernels are SM100-only and **silently emit garbage on SM121** —
which is GB10, i.e. this hardware. Reported by
[Death-By-Tokens](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark)
against SGLang, along with a second path to the same `!!!!` signature: a `_compact_kv` Triton
kernel that does not actually compact, leaving uninitialised NaN holes where interleaved `-1`
top-k indices should have been squeezed out.

**Why it is worth knowing even on vLLM.** "SM121 silently emits garbage" is a *class* of bug on
this hardware, not a single instance. Note the different signature from A1: `!!!!` is NaN
collapse, whereas a corrupt shard stays fluent.

### A4. Reasoning trace leaks into `content` **[us]**

**Signature.** Output is correct but prefixed with the model's internal reasoning and a stray
`</think>`, with no opening tag.

**Fix.** `--reasoning-parser qwen3`. Cosmetic, but it will corrupt any automated scoring that
reads `content` naively.

---

## B. It will not load

### B1. `ValueError: no module or parameter named 'ngram_embedding.weight_scale'` **[us]**

**Cause.** `_get_ple_embedding_quant_method()` in
`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` gates on `isinstance(quant_config,
Fp8Config)`. `RadixArk/Qwen3.8-Flash-Next-NVFP4` ships the PLE in *exactly* the format that
method implements — F8_E4M3 shards plus one global BF16 `ngram_embedding.weight_scale` — but the
**body** is NVFP4, so `quant_config` is `modelopt_fp4` and the gate rejects it. The embedding is
then built unquantized, and loading dies on the scale it never expected to see.

**Fix.** Accept `modelopt` / `modelopt_fp4` (see `scripts/apply-pr53896.sh`), and heed A2.

**Consequence for the field:** RadixArk NVFP4 **does** load on vLLM. Checkpoint tables listing it
as incompatible are wrong; it needs a one-line change.

### B1b. DeepGEMM faults on sm_121 with blockwise-FP8 weights **[us + field]**

**Signature.** The engine dies during the startup profile run, before serving anything:

```
torch.AcceleratorError: CUDA error: unspecified launch failure
RuntimeError: CUDA driver error (deepgemm-src/.../jit/handle.hpp:154): 719 CUDA_ERROR_LAUNCH_FAILED
  vllm/utils/deep_gemm.py:464 in fp8_gemm_nt
  vllm/v1/worker/gpu/model_runner.py:854 in profile_run
```

**Workaround that works today:** `VLLM_USE_DEEP_GEMM=0`, which falls back to
`CutlassFp8BlockScaledMMKernel`. Everything we measured is on that fallback, so our decode figures
may be a floor rather than a ceiling.

**Cause — our first published explanation was wrong.** We reported it as a too-coarse capability
gate (`support_deep_gemm()` accepts the whole `120` family, and GB10 is sm_121) and filed
[vllm#54125](https://github.com/vllm-project/vllm/issues/54125) on that basis. `jahnclawdmonet`
ran it down on the same hardware and found an **attribute-name mismatch**: this class stores its
kernel as `self.w8a8_block_fp8_linear`, while `process_weights_after_loading` guards on
`hasattr(self, "fp8_linear")` — the name every *other* method in that file uses. The guard never
fires, so the kernel's own weight post-processing (**UE8M0 requantization and int32 packing of the
block scales**) is silently skipped.

We verified the mismatch in our tree (assignment at line 784, guard at 824) and fixed it. **It did
not resolve the crash**, which is informative: the failure is at
`_fp8_gemm_nt_impl(..., disable_ue8m0_cast=not use_ue8m0, ...)`. On sm_121
`is_deep_gemm_e8m0_used()` returns True through the same family-120 check, so vLLM selects an
**E8M0 kernel variant whose scale format the checkpoint does not supply** — plain FP32 block
scales. Restoring the guard cannot retroactively convert scales that were never written in that
format.

So the accurate statement is narrower than either of the first two: the gate is not obviously
wrong, the attribute bug is real but not sufficient on its own, and the operative mismatch is
between the **E8M0 kernel variant selected on sm_121** and the scale format of a checkpoint
produced elsewhere.

### B2. `pidfd_getfd: Operation not permitted` in the PLE offload worker **[us]**

**Signature.** Both workers load all 206 shards successfully — this takes ~10 minutes — and
*then* the engine dies. What the API server prints is useless:

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

The actual cause is further up the log, in the `PleOffloadWorker` stream:

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

**Cause: `kernel.yama.ptrace_scope = 1`** — the default on Ubuntu and DGX OS. It restricts
`PTRACE_MODE_ATTACH`, which `pidfd_getfd` requires, to **descendants only**. `PleOffloadWorker`
and the GPU worker are *siblings* (both children of the engine), so neither may attach to the
other and the CUDA-IPC tensor handoff is refused.

> We first published this as a Docker seccomp restriction. That was wrong, and we only found out
> by building the same stack **bare metal**, where it failed identically. `CAP_SYS_PTRACE`
> bypasses yama, which is why the Docker flag fixed it — for the wrong reason. Corrected
> upstream.

**Fix, by deployment:**

| deployment | fix |
|---|---|
| Docker | `--cap-add=SYS_PTRACE` |
| bare metal / systemd | `AmbientCapabilities=CAP_SYS_PTRACE` on the unit |
| bare metal / shell | inherits the login's capabilities; usually already works |

For systemd, `cap_sys_ptrace` in `CapabilityBoundingSet` is **not sufficient** — that only bounds
what *may* be held. A `User=` service holds no effective capabilities without:

```ini
[Service]
AmbientCapabilities=CAP_SYS_PTRACE
```

`sysctl -w kernel.yama.ptrace_scope=0` also works but weakens ptrace machine-wide; the ambient
capability is scoped to one service and is the better answer.

This affects anyone using vLLM's official `VLLM_PLE_CPU_OFFLOAD`, in a container or not. It does
**not** affect the mmap-hook approach (single process, no IPC handoff).

### B2b. `OSError: Could not load this library: libtorchcodec_image.so` at startup **[us]**

**Signature.** The server dies at import, before loading anything, on a host without system
ffmpeg.

**Cause.** `vllm/multimodal/video.py` guards the torchcodec import as
`except (ImportError, RuntimeError)`, but torchcodec raises **`OSError`** when its shared library
cannot load. `vllm/multimodal/__init__.py` pulls this in unconditionally, so an
**installed-but-unloadable** torchcodec is strictly worse than an absent one — absence raises
`ImportError`, which *is* caught and falls back to a `PlaceholderModule`.

**Fix.** Either install system ffmpeg (`apt install ffmpeg`), or **do not install torchcodec**.
It is genuinely optional — vLLM has `check_torchcodec_available()` — and only video input needs
it; text and image are unaffected.

### B3. `ImportError: cannot import name 'checkpoint_has_lm_head'` **[method]**

**Cause.** A venv with mixed vLLM versions — a partial upgrade, or files bind-mounted/copied
between builds. Ours accumulated this over a day of patching.

**Fix.** Rebuild the venv, or use the container. Note the related trap from earlier work on this
box: `cp -a` of a venv keeps the **original's** interpreter in every shebang, so the copy silently
runs the wrong Python. Prove which interpreter is live with a log line from inside the test code,
not by inspecting paths.

### B4. MLIR "weakly congruent" crash at boot **[field]**

TMA-O enabled for varlen, where the ragged epilogue is rank-broken. Reported by Death-By-Tokens
against SGLang; the correct guard survives in a comment one line above the bug.

### B5. `Unsupported rhs dtype fp8e4nv` on long prompts **[field]**

The long-prefill sparse kernel feeds fp8-loaded K straight into `tl.dot`. Compiles fine on short
prompts, then kills the server on the first ~100k+ request. Death-By-Tokens, SGLang.

---

## C. It hangs

### C1. Silent hang at TP=1 **[us, fixed upstream]**

**Cause.** Uniproc executor selection. **Fix:** `--distributed-executor-backend mp`.
[vllm#53960](https://github.com/vllm-project/vllm/issues/53960), credited to dolf3131; fixed
upstream in `95dc96d1d012`.

### C2. Box-wide OOM during JIT or compile **[us, prior]**

On GB10 the CPU and GPU share one 128 GB pool, so an unbounded JIT fan-out takes down the whole
machine, not just the server — SSH included. Bound it:

```bash
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1
```

A driver upgrade invalidates the FlashInfer JIT cache, so the first launch afterwards is exactly
when this fires. Warm the cache at low `gpu-memory-utilization` after any driver change.

---

## D. Measurement traps — you get a number, and it is a lie

These produced **published, wrong conclusions** in this project. They are worse than crashes,
because nothing tells you they happened.

### D1. Sampling the warmup forward pass **[method]**

vLLM runs a dummy forward at startup whose buffers are deliberately zeroed. We instrumented a
tensor, read all zeros, and reported it as a finding. **Instrument by call index** and skip the
first few.

### D2. `print()` inside a compiled region **[method]**

`Dynamo does not know how to trace builtin operator print`. The debug statement changes the
compilation path, so the experiment measures something other than the configuration under test.
Use `logging`.

### D3. Content-checking at offset 0 **[method]**

See A1. Row 0 of a corrupt file is very often intact, because truncation damages the tail.

### D4b. Published quality metrics copied between checkpoint variants **[field]**

**Signature.** A model card carries `gsm8k_metrics.json` / `aime26_metrics.json` with strong
scores, and you use them to choose between variants.

**Check the hash, not the score.** Across four HuggingFace repos — a plain NVFP4 build, an FP8
dense-quantized fork of it, and a 512→448 expert-**pruned** variant — those files are
byte-identical:

```
gsm8k  sha256 88766f7e…  score 0.9727  latency_seconds 829.4909560070373
aime26 sha256 eb4acd8c…  score 0.9875  latency_seconds 11196.289524045998
```

Identical latency to ten decimal places across models with different weights is not a coincidence,
and the `model` field inside the files names the *original* build. A pruned model reporting its
unpruned parent's numbers is the tell.

**How to check, in seconds:**

```bash
for r in RepoA RepoB; do
  curl -sL "https://huggingface.co/$r/resolve/main/gsm8k_metrics.json" | sha256sum
done   # identical hashes across differing weights = copied, not measured
```

Also read the `model` / `base_url` fields inside — they frequently name a different checkpoint
than the repo you found them in. Treat any metric you did not generate as provenance, not
evidence, and re-measure on your own workload.

### D4. `usage.prompt_tokens_details.cached_tokens` is inert **[us, prior]**

It reports 0 even on confirmed prefix-cache hits. Measure `vllm:prefix_cache_hits_total` deltas
from `/metrics` instead. We first concluded "the cache never hits" on both arms of a comparison.

### D5. Prefix cache does not hit until the *second* repetition **[us, prior]**

A two-request probe measures a defect that is not there. Use three.

### D6. `temperature: 0` is not reproducible under concurrent load **[us, prior]**

Concurrent requests change the output (~0.873 self-similarity) with no speculation involved. Any
comparison resting on text identity is invalid under load.

### D7. `pkill -f 'bin/vllm serve'` matches its own shell **[method]**

It kills the launcher before it can relaunch. Use PIDs.

### D8. Publishing before the controls are run **[method]**

We published "the body is the suspect" and were refuted within hours by someone serving the same
body coherently. Two upstream issues had to be retracted. The premise was corrupt weights (A1) —
but the process failure was reporting a conclusion whose control had not yet been run.

---

## Verification checklist

Before you conclude *anything* about a model, quantization, or kernel on this hardware:

1. `sha256sum -c SHA256SUMS` against HF's `lfs.sha256`. Not sizes. Not "the download said OK".
2. Confirm which interpreter and which vLLM version is live, from inside the running process.
3. Discard the first forward passes before reading any instrumented tensor.
4. For cache or speed claims, use at least three requests and read `/metrics`, not `usage`.
5. If a symptom is invariant to every configuration you change, go back to step 1.

## The Flash-Next server is on :8092, not :8080

`serve-fnext.sh` binds `--port 8092`. The rest of the fleet is on :8080, so every health check
and bench invocation written from muscle memory hits the wrong port and returns `000` — a
*connection* failure, which looks identical to "still warming up" in a readiness loop.

This burned two 25-minute monitors and one benchmark run that reported "NEVER READY" against a
server that had been serving for ten minutes. `curl` returning `000` rather than `401`/`404` is
the tell: nothing is listening there at all.

Check `ss -ltn | grep 809` before concluding a startup has stalled.

## `..` in a curl path strips one segment, it does not climb to the root

A health probe written as `$URL/../models` where `$URL` ends in `/v1/chat/completions` does **not**
resolve to `/v1/models`. curl normalizes it to **`/v1/chat/models`** — `..` removes the last
segment (`completions`), leaving `/v1/chat/`, and `models` is appended there. That 404s forever.

```
$ curl -v 'http://127.0.0.1:8092/v1/chat/completions/../models'
> GET /v1/chat/models HTTP/1.1     # not /v1/models
```

This ran two 20-minute benchmark arms against a **healthy** server and reported both as
"never ready". The failure is silent and indistinguishable from a slow start, because a
readiness loop cannot tell a 404 from a server that has not finished loading unless it
looks at the code — and this loop only tested `= 200`.

Write health URLs out in full. Never construct them relationally. If a readiness loop times
out, `curl -v` the exact URL it used before concluding anything about the server.

## Startup is dominated by weight loading, not compile

`Loading weights took 556.07 seconds` — 9.3 minutes, against 44 s of torch.compile with cache
hits. Every restart in an A/B costs ~11 minutes wall before the first request. Size readiness
windows at 30 minutes, and prefer arms that avoid a restart.

## "The arms disagree, and I cannot tell which one to believe"

Three distinct harness defects produced this on 2026-08-31, all invisible in the results file.

### The metric timed unequal work

`agentloop.py` sent `max_tokens: 130` with no `ignore_eos` and never recorded
`completion_tokens`. **`max_tokens` is a ceiling, not a target**, so each arm was timed on however
many tokens the model happened to emit — an arm whose turns stopped at 40 tokens beat one whose
turns ran to 130 with no difference in speed.

**Symptom:** identical configs spanning 36% (1.94 then 1.43 s/turn), while decode on the same arms
was reproducible to 1%. **Tell:** the noise is in one column only.
**Fix:** `ignore_eos`, record per-turn tokens, refuse the verdict unless the total is exactly
`turns × max_tokens`, and report `ms/tok`. **Validate the fix**: `ms/tok` must agree with
`1000/decode_tps` from an independent benchmark — it now does, to ~1%, and the reproducibility
went from 36% to **0.4%**.

### Benchmarks serialized by unit name instead of by resource

Each queued wave waited on a *named* systemd unit (`wave3` waits for `wave2`, …). That holds only
while nothing is ever stopped. Stopping one unit made the next fall through **immediately** and
load a second vLLM server onto a busy GPU.

**Symptom:** an arm whose last turns step off a flat plateau (5.2-5.8 s, then 7.5 s), or arms dying
with *"Free memory on device is less than desired GPU memory utilization"*. **Tell:** compare the
arm's turn timings against the wall-clock of the next log file's creation.
**Fix:** wait for the **resource**, not the unit — no `vllm` process and no arm unit — or better,
run every arm from a single unit, which cannot race itself. Cost of learning this: one contaminated
arm, five dead arms, ~30 minutes of GPU time.

### systemd `Environment=` splits on spaces — 7 arms lost in one batch **[method]**

**Signature.** An arm dies in ~10 s with an **empty** log. `systemd-run` reports
`Failed to find executable all: No such file or directory`.

**Cause.** `${10:+--property=Environment=FN_EXTRA=${10}}` is unquoted, so bash word-splits a value
like `--mamba-cache-mode all` into two argv entries; `systemd-run` then takes `all` as the start of
the command. `serve-fnext.sh` never runs, hence the empty log. This is the same gotcha already
recorded for `Environment=` quoting, met from the other side.

**Cost.** `GENBIS` plus all six `M_all_*` / `M_align_*` arms on 2026-09-01 — the whole
`mamba_cache_mode` separation and the decode-side bisection — ~1.5 h of queue time, discovered
only because the watchdog read the death cause instead of the verdict line.

**Fix, verified with a non-GPU unit before re-running anything:** build the property as ONE argv
entry with systemd-level quoting inside the value:

    EXTRA_PROP=()
    [ -n "${10:-}" ] && EXTRA_PROP=(--property="Environment=\"FN_EXTRA=${10}\"")
    systemd-run ... ${EXTRA_PROP[@]+"${EXTRA_PROP[@]}"} ...

Delivered value checked: `FN_EXTRA=[--mamba-cache-mode all]`, wordcount=2.

**Rule.** Any `Environment=` value that can contain a space goes through an array with quoting
inside the value, never through `${var:+...}`. And a 10-second death with an empty log is a
launcher failure, not a model failure — read `systemd-run`'s own stderr, not the arm log.

### An experiment hook left unconditional ran in every arm for a day **[method]**

**Signature.** `qsa.py:839 blocks.sort(dim=1).values  # vllm#54521 determinism fix` — added as one
of three top-k experiment hooks, but unlike the other two it was **not env-gated**. Every arm from
that point ran a non-stock top-k path, and `torch.topk` vs stock then compared *sort+torch* against
*sort+persistent*, not the two kernels. When `visible < k` (the 60-token probe) the sort can move
uninitialised padding below the real indices where the expansion kernel reads it. Whether that is
what produced the `'#'` vs `'The'` top-token split is now `topk_boundary.py` T5/T6.

**How it was found.** Not from memory — the audit that caught it diffs every `vllm/**/*.py` against
the sha256 in pip's `RECORD`, then greps the differing files for
`EXPERIMENT|PROBE|DIAGNOSTIC|_dq_calls|# temporary|\.item\(\)`. Ten files differed; seven were
deliberate patches, three edits were unwanted (the sort, two inert hooks, and a bare `print("PROBE
lmhead ...")` in `model.py` left over from the lm_head work). `tools/determinism/cleanup_hooks.py`
removes them from a known backup and verifies the result differs from stock only in FP8-KV lines.

**Rules.**
- Every experiment hook is env-gated, no exceptions. An unconditional "fix" is a config change.
- Before any batch, run the RECORD diff. Backups beside sources (`*.py.pre*`) are a *list of edits
  made*, not proof they were reverted.
- A "clean" venv claim needs the diff output, not the word.

### A scratchpad file named like a package torch imports broke `import torch` **[method]**

**Signature.** `RuntimeError: generic_type: cannot initialize type "GradBucket": an object with
that name is already defined`, from `torch/__init__.py` at `_C._initExtension`, on a plain
`import torch`. Works from `/`, fails from the scratchpad. `python -P` (safe path) fixes it.

**Cause.** Python prepends the *script's directory* to `sys.path`. Torch's init does an optional
`import cuda` (the cuda-python package); a scratchpad file `cuda.py` won, imported torch itself,
and re-entered initialisation. **Two wrong diagnoses preceded the right one**, both recorded here
because each looked complete: a stray partial vLLM checkout (`vllm/`, 490 files, from a research
subagent) and a `trace.py` shadowing stdlib `trace` — both real collisions, neither the cause.
Renaming each "fixed" nothing.

**Finding it — the method that worked.** `python -v -c "import torch" | grep scratchpad/` lists
exactly the files the interpreter loads from the scratchpad. One line: `cuda.py`. The enumerator I
wrote first (names vs `importlib.util.find_spec`) *excluded* the case where the scratchpad file wins
resolution, which is the shadow case — it could only ever find benign collisions. Ask the
interpreter what it loaded; do not reason about what it might.

**Rules.**
- Run every instrument with `PYTHONSAFEPATH=1` (or `python -P`). The runners now export it.
- Never name a scratch file after a module *or an optional dependency*: `cuda`, `trace`, `types`,
  `test`, `token`, `code`, `io`, `select`, `signal`, `profile`, `queue`, `random`, `secrets`.
- "Works from another directory" is the tell. Then `-v`, not guesswork.

### A stray installer line survived six script derivations **[method]**

**Signature.** An arm's output shows `layer-hash diagnostic INSTALLED` followed by `already
installed`; the installed hook is the *narrow* one, and the widened submodule hook never takes.

**Cause.** Every runner was derived by replacing text between a header anchor and `ALL DONE`. A
`$PYBIN $S/layerhash_patch.py` install (with its "(BISECTION…)" echo) from the original
`bisect.sh` sat *above* the first anchor and was inherited unchanged by rerun → gdncuda →
statehash → plehash → layer0sub. It installed the narrow hook first; the intended installer saw
the marker and skipped. For arms that didn't set `VLLM_LAYER_HASH` it was inert; for `LAYER0SUB`
it silently substituted the wrong instrument. Cost: one 13-minute arm, caught by counting hooked
modules (49, the narrow count) rather than trusting "INSTALLED".

**Rules.**
- Derive runners from a clean template, not by chained text replacement; or diff the head of every
  derived script against its parent before launch.
- Every instrument arm must log *what it hooked* (module count / names), and the analysis must
  check that count against the expectation before reading a single hash.
- A patch installer that finds its marker already present must say **which version** is there.

### Checks that cannot fail

- `sudo -n -u llm test -r FILE` reports "cannot read" when it merely lacks a password. It produced
  a confident wrong diagnosis of a permission problem that did not exist.
- `cmd | head -3 || echo clean` never prints `clean`: `head` exits 0. A secret scan written this way
  passes unconditionally.
- A filtered monitor goes **silent** when a run dies unexpectedly, and silence reads as "still
  running". Heartbeats must report **unconditionally**, and should assert the invariant — ours now
  prints a warning if more than one vLLM server is alive.

The thread joining all three: **a guard written in a note does not travel to the next script.** Each
of these was already documented here before it recurred. Put the assertion in the tool
(`tools/agentloop.py`), not in the prose.

## "The same config gives two different answers, run to run"

Observed 2026-08-31: `ms/tok` for a fixed configuration clusters into two modes, ~32 and ~48+.
k=3 measured 32.48 then 49.08 (51% apart); k=4 gave 31.93 / 38.96; n=6 gave 52.11 / 71.85. Meanwhile
n=5 (31.81 / 31.97) and no-spec (43.37 / 43.64 / 43.80) replicate to under 1%. So it is not the
harness and not the box drifting — the same code takes one of two paths depending on the server
start.

⚠️ **SUPERSEDED a third time — there is no n=6 anomaly. MTP is unstable at every depth.**

The earlier readings (instrumentation → subagent page-cache contamination → "it was both") were all
attempts to explain why *n=6 specifically* was slow. Pooling all 66 recorded arms on 2026-09-01
shows the premise was wrong. Grouping by drafter and measuring the spread **within one config
across identical restarts**:

| drafter | repeated configs | worst within-config spread | overall range (ms/tok) |
| --- | --- | --- | --- |
| no-spec | 5 | **1.10×** (four of five are 1.00–1.01×) | 42.8 – 47.7 |
| ngram / ngram_gpu | 2 | **1.09×** | 28.5 – 33.8 |
| **mtp** | 5 | **1.83×** | **31.5 – 77.5** |

Per-config, MTP: `MTP2` 47.8→77.5 (1.62×), `MTP3` 32.5→49.1 (1.51×), `MTP4` 31.5→57.6 (1.83×),
`MTP6` 52.1→74.7 (1.43×), `MTP7` 56.7→57.2 (1.01×). The flip is **per server start** — the same
binary, flags and prompts land in a fast regime or a slow one, and `MTP7` shows it can land the
same way twice, so a 2-sample agreement proves nothing.

**Consequences, all of them ours to own:**

1. **The n=6 attribution is withdrawn.** The "2.2× from debug code" (33.3 vs 73.5) was two n=1
   samples of a quantity whose own within-config spread reaches 1.83×. The debug blocks were real
   and worth removing, but the effect attributed to them is **not established** and never was.
2. **Every MTP comparison at n≤3 is underpowered.** The sort-fix, row-cap and `torch.topk` arms
   (`P1` 1.84×, `P2` 1.39×, `R192` 1.95×, `TG` 1.76×) sit on an instrument with ~1.8× noise. None
   of them could resolve an effect smaller than about 2×. Their *categorical* results
   (3 distinct outputs of 3) are unaffected — that is a text-identity test, not a timing one.
3. **The stable configs are stable.** no-spec at 43.4–43.8 across four starts, ngram at 28.5–29.3,
   `B16384` at 45.4/45.5, `CG_PIECE` at 43.6/43.8. So this is not box-wide noise, not thermal,
   and not the harness — those would move every arm. It is confined to the MTP path.

**What it points at.** MTP adds a drafter forward and a rejection sampler that no-spec and ngram do
not have, and `ms/tok` was already shown to be governed by mean accepted length. So the likely
reading is that **MTP acceptance itself varies per start**. That is consistent with — and may be
the same defect as — the per-request prefill divergence found by the logit probe on 2026-09-01
(`notes/…` determinism thread): if the drafter's own forward pass is affected, acceptance moves,
and `ms/tok` follows.

**Not yet established:** that the timing instability and the text divergence are one defect. They
are correlated in mechanism and were found in the same week; that is a hypothesis, not a result.

**Method rule this cost us:** never report an MTP timing from a single start. Minimum three starts
per config, report the range, and treat any difference under 2× as unresolved.

---

Previous reading, kept for the record:

⚠️ **RESOLVED — it was BOTH, and they hit different configs.** First read: the instrumentation.
Then: research subagents evicting the page cache ([[read-only-is-not-load-free]]), confirmed by a
quiet re-run returning 31.47 against a contaminated 57.61 — and I wrote here that the debug code was
*not* the explanation. **That correction was itself too broad.** After the debug blocks were removed
on 2026-09-01, **n=6 measured 33.3 ms/tok against 73.5 on a quiet box with them present** — a 2.2×
effect from our own instrumentation, on that config.

So: **subagent contamination inflated k=4 and friends; the debug counter cost n=6 a factor of two.**
Two independent self-inflicted effects, overlapping in time, each mistaken for the other and each
briefly mistaken for a property of the model. Why the counter hit n=6 and not k=3/k=4/n=5 — which
measured ~32 *with* it present — is unexplained; a per-forward attribute mutation on a compiled path
can change guard/recompile behaviour per graph, but that is a hypothesis.

**Consequence: every depth number collected before 2026-09-01 is suspect** and is being re-measured.

**First candidate (superseded).** Two debug blocks were still live in the production venv:

- `models/…/nvidia/ple_layer.py:669` — `self._dq_calls = getattr(self, "_dq_calls", 0) + 1` on
  **every PLE forward**, and on calls 6-9 `_e.abs().max().item()` and `(_e != 0).sum().item()`:
  **device syncs on the hot path**, plus a Python attribute mutation inside a region that may be
  compiled.
- `model_executor/layers/quantization/modelopt.py:2501` — `# PROBE (temporary)` `logger.warning`
  firing for every `lm_head` prefix.

A per-forward attribute mutation on a `torch.compile`d path can force a graph break or a guard
failure, and *whether* it does can depend on when compilation happened relative to those calls —
which is a mechanism that produces a per-start coin flip. **Candidate, not established:** it has not
been shown that this code sits inside a compiled region here.

**Test, in order:** finish the running queue (changing the code mid-run makes the remaining arms
incomparable), remove both blocks, then re-run the matched pair that showed both modes — k=3, twice.
If the bimodality disappears, a large part of the depth analysis was measuring instrumentation.

This is the second time in one session that debug code left in the tree has mattered, and the patch
MANIFEST already records a *first* occurrence — *"a stray `print` in `get_quant_method` was still
firing in production"*. **Grep the venv for probes before trusting a benchmark**, not after:

    grep -rnE "PROBE|_dq_calls|# temporary|\.item\(\)" $SP/vllm/models/qwen3_8_flash_next/ \
        $SP/vllm/model_executor/layers/quantization/modelopt.py
