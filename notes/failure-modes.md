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
