# Reproducing this on your own DGX Spark

A working server, start to finish. The [README](README.md) says *what we measured*; this says
*what to type*. Every version pin below is load-bearing — three of them have a documented failure
mode if you take the newer thing.

**Target:** `36.5 tok/s` single-stream, ~`100 tok/s` aggregate at 16 concurrent, 32K+ context,
tool calling, on one GB10 with 128 GB unified memory.

**Time:** ~5 h, almost all of it downloading. Budget ~3 h for the weights alone.

---

## 0. What you need

| | |
|---|---|
| hardware | NVIDIA DGX Spark (GB10, **sm_121**, 128 GB unified, aarch64) |
| disk | **~140 GB free.** The base is 123 GB and the overlay adds ~12 GB |
| vLLM | `0.1.dev20073+g8e685d198` — the **#53896 + #53899 preview build** |
| FlashInfer | **0.6.17**, pinned |
| torch | `2.13.0+cu130` |

### Two pins that will bite you if you "upgrade"

- **Do not build from vLLM `main`, even though #53896 merged on 2026-08-31.** `main` has the model
  but **not** PLE offload — #53899 is still open and `vllm/v1/ple_offload/` does not exist there.
  Without it the 51.2 B-parameter n-gram table stays resident on the GPU and the model does not fit.
  The two trees have diverged (#53899 is not stacked on the merge), so this is a real port, not a
  cherry-pick. Wait for #53899, or build from its branch head.
- **Do not take FlashInfer 0.6.18.** flashinfer#4757 removed **SM121a from the aarch64 JIT-cache
  arch list** and it was cherry-picked into the 0.6.18 release. You lose the prebuilt cubins, fall
  back to runtime JIT, and the unbounded ninja fan-out can take the whole box into a global OOM —
  not just vLLM. vllm#54313 bumped vLLM's pin to 0.6.18 on 2026-08-30, so **re-pin 0.6.17 after any
  vLLM change**.

  If you are ever forced onto 0.6.18, it is survivable but do it deliberately: flashinfer#3170's
  audit notes `compute_120f` covers both CC 12.0 and 12.1, and the arch-specific `sm_121a`
  requirement applies **only to sparse MMA** (`mma.sp .kind::mxf4nvf4`) — which this model does not
  use; our `mxf4nvf4` references are dense `tcgen05.mma` in the NVFP4 MoE mainloop. So the loss is
  bounded. Warm the JIT cache **at a low `--gpu-memory-utilization` first**, with `MAX_JOBS=2` and
  `FLASHINFER_NVCC_THREADS=1` set (both are already in the launcher). A cold JIT rebuild at
  util 0.90 is how this box was taken down once.

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

Eight local vLLM patches. `apply.sh` is idempotent and reports `applied` / `already` / `FAILED`:

```bash
cd patches && ./apply.sh
```

Order matters and the script hardcodes it — `hyperconnection.py` **must** precede `model.py`,
because `model.py` passes `quant_config=` to `GatedResidual` and upstream's signature does not
accept it. See [MANIFEST](patches/MANIFEST.md) for what each one does and why.

**Any `pip install`/upgrade of vLLM in this venv silently reverts all eight.** The symptoms are
non-obvious — a startup hang at `warmup_kernels`, HTTP 400 on every tool call, missing scale
parameters. Re-run `apply.sh` after any reinstall.

## 4. Serve

[`scripts/serve-flashnext.sh`](scripts/serve-flashnext.sh) is the live launcher from our box.
The flags that are not obvious:

| setting | why |
|---|---|
| `VLLM_PLE_CPU_OFFLOAD=1` | the whole point; keeps the 51.2 B n-gram table off the GPU |
| `VLLM_USE_DEEP_GEMM=0` | DeepGEMM gates on device-capability *family* 120, which sm_121 satisfies — then faults with `unspecified launch failure` (vllm#54125) |
| `VLLM_GDN_DECODE_KERNEL=triton` | the default CUDA kernel deterministically hangs the engine at c≈32 with FP8 GDN projections. No error, requests just stall |
| `CUTE_DSL_ARCH=sm_121a` | required for the FlashInfer CuteDSL path |
| `--max-model-len 32768` | **not 8192.** A single code task emitted 31,115 characters of *thinking* before 12,931 of content. 8192 cannot hold this model's own reasoning |
| `--max-num-seqs 16` | **not 2.** Our early "concurrency ceiling" was this flag, not the hardware — the box reaches 266.8 tok/s at 48 streams |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | without these, every request carrying `tools` returns **HTTP 400** |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | +35%. `k=5` hard-fails (QSA ring capacity must divide the attention block size) |

> ⚠️ **Never combine MTP with `--async-scheduling`.** `_prepare_ngram_context` reads the CPU token
> mirror while it still holds speculation's `-1` placeholders, so the n-gram context is wrong.
> Silently. No benchmark reveals it.

If you serve in Docker you also need **`--cap-add=SYS_PTRACE`**: PLE offload's `rebuild_cuda_tensor`
needs `pidfd_getfd`, and without it the engine dies ~10 minutes in with only `Failed core proc(s): {}`.

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
  capability-*family* check excludes all of 12.x), so every request takes
  `torch.ops._C.persistent_topk`, which vllm#51782 reports silently returns wrong values in a
  data-dependent way. Four other explanations were tested and eliminated
  ([write-up](notes/temp0-nondeterminism.md)).
- **Long-prompt hangs (>8k)** are `is_arch_support_pdl()` returning True for anything with
  `major >= 9`, so PDL is used in `_build_qsa_metadata_kernel` where the dependent kernel waits
  forever (vllm#53960).
