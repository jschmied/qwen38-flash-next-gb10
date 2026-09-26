# Reproducing this on your own DGX Spark

A working server, start to finish. The [README](README.md) says *what we measured*; this says
*what to type*. Most pins below are load-bearing and have a documented failure mode.

**Target** (the prod configuration, 2026-09-26): **46.5 tok/s** single-stream, ~**100 tok/s** aggregate
at 4 streams, 1.10 s per agent turn, 32K context, tool calling, on one GB10 with 128 GB unified memory.

**Time:** ~5 h, almost all of it downloading. Budget ~3 h for the weights alone.

> **This recipe changed on 2026-09-23.** It used to be nightly `dev524` plus a hand-port of the
> PLE-offload branch #53899, which needed 64 GiB of swap and `CAP_SYS_PTRACE`. That stack is retired:
> the PLE table is now read in place from the checkpoint files ([vllm#58439](https://github.com/vllm-project/vllm/pull/58439)).
> Neither the swap nor the capability is needed any more. The old recipe is in this file's history
> (commit `d04c85d`), if you need it.

---

## 0. What you need

| | |
|---|---|
| hardware | NVIDIA DGX Spark (GB10, **sm_121**, 128 GB unified, aarch64) |
| disk | **~140 GB free.** The base checkpoint is 126 GB; the derived one is a hardlink overlay, so only the changed tensors take new space |
| swap | Ubuntu's default 16 GiB is enough; the mapped PLE path uses 5–6 GiB of it |
| vLLM | nightly `main` **`1ea7c63f4`** (`0.29.1rc1.dev533+g1ea7c63f4`) + our overlay (§3) |
| FlashInfer | `0.6.18.post1` — python + cubin + jit-cache, all three at the same version |
| torch | `2.13.0+cu130` |

### Take FlashInfer 0.6.18, and keep the three packages at one version

Neither 0.6.17 nor 0.6.18 ships an sm121 artifact, and none is needed: `compute_120f` covers CC 12.0
**and** 12.1, and `sm_121a` is required only for sparse MMA, which this model does not use. After the
cut-over, `~/.cache/flashinfer/0.6.18.post1/121a/` holds 0 modules and a full start log has no
`ninja`/`nvcc` lines ([det-208](notes/determinism-investigation.md)). Sources:
`flashinfer-python` from PyPI, `flashinfer-cubin` from `https://flashinfer.ai/whl/` (PyPI tops out at
0.6.13), `flashinfer-jit-cache` from `https://flashinfer.ai/whl/cu130/`. `jit/env.py` asserts
cubin == python and aborts the import on a mismatch.

> **Startup takes ~12 minutes.** Loading ~120 GB dominates. Don't mistake the wait for a hang.

---

## 1. Weights

Start from `RadixArk/Qwen3.8-Flash-Next-NVFP4` (125.9 GiB): a main model (78.2 GiB, 196 files) and the
PLE n-gram table (47.7 GiB, 10 `model-plefp8-*` files, FP8).

```bash
# ~3 h. Cap the rate; the Xet CDN behaves badly with many connections.
aria2c -x1 -s1 --max-overall-download-limit=6M -i urls.txt
```

> ⚠️ **Verify `lfs.sha256`, never file size.** `aria2` preallocates, so a file reaches its final
> size the instant it starts. Two size-correct, byte-corrupt shards here produced *fluent garbage*
> that was invariant to every configuration change — it cost a full day and two retracted upstream
> issues. See [fetching a slice](notes/fetching-a-slice.md).

## 2. The derived checkpoint

The published checkpoint leaves the dense projections in BF16 and the MTP drafter in BF16. Four changes,
none of which costs measurable quality:

1. **dense projections → FP8** (from `lovedheart`'s mixed checkpoint): same size, **+39 %**
   ([fp8 checkpoint](notes/fp8-mixed-checkpoint.md))
2. **`lm_head` → FP8** (ours): **+11 %**, +19 % once MTP is on
   (`scripts/quant_lmhead.py`, [quantizing lm_head](notes/quantizing-lm-head.md))
3. **MTP drafter experts → NVFP4** (ours): acceptance unchanged, **+26 % KV capacity**, and it lets the
   drafter take the quantized MoE path (`scripts/quant_mtp_nvfp4.py`,
   [quantizing the MTP drafter](notes/quantizing-the-mtp-drafter.md)). This gives `qwen38-flash-next-mtpfp4`.
4. **A 32k draft vocabulary** for the drafter's argmax: `tools/draft_vocab/draft_vocab_32768.txt`
   (token ids, 99.6 % coverage of our agent output, +6.4 % single-stream, acceptance unchanged).
   `tools/draft_vocab/build_vocab.py` rebuilds it from your own traffic.

Build 1–3 as a **hardlink overlay**, not a copy: only the changed tensors are new files.

> ⚠️ **A hardlinked `config.json` is shared, and editing it in place edits the base too.** This
> silently corrupted our production checkpoint once. Break the link before editing:
> `cp config.json config.json.new && mv config.json.new config.json`.

> ⚠️ **The runtime reads `config.json`'s embedded `quantization_config`, not `hf_quant_config.json`.**
> Under `quant_algo: MIXED_PRECISION` the authoritative field is **`quantized_layers`**, not
> `config_groups`. Writing the wrong one gives you a W4A4 kernel with an uninitialised activation
> scale: random logits, immediate end token, **zero characters of output, no error**. Gate every
> build offline before starting a server ([choosing a quant scheme](notes/choosing-a-quant-scheme.md)).

> **Current `main` cannot load these derived checkpoints.** Its new `MergedColumnParallelLinear.load_weights`
> falls back to the module itself for keys it does not know, and fails with
> `'MergedColumnParallelLinear' object has no attribute 'data'` (seen on `378504a54`, 2026-09-26).
> Stay on `1ea7c63f4` until that is resolved.

## 3. The venv: nightly wheel + one overlay

**Clone an existing venv that already has `torch 2.13.0+cu130` and FlashInfer 0.6.18.post1; do not
build a fresh one.** A fresh venv breaks the torch pin ([build recipe](tools/main/BUILD-RECIPE.md)). A
`cp -a` copy keeps the original's interpreter path in its script shebangs, so fix them:

```bash
O=/opt/llm/runtime/vllm-venv-<existing>; N=/opt/llm/runtime/vllm-venv-main1ea7
cp -a $O $N
grep -rlI "^#!$O/bin/" $N/bin | xargs -r sed -i "1s|^#!$O/bin/|#!$N/bin/|"
SP=$N/lib/python3.12/site-packages
rm -rf $SP/vllm $SP/vllm-*.dist-info
SHA=1ea7c63f4af7bb4fd6f025c8db44434ab274cb51
$N/bin/python -m pip install --no-deps \
  "https://wheels.vllm.ai/$SHA/vllm-0.29.1rc1.dev533%2Bg1ea7c63f4-cp38-abi3-manylinux_2_28_aarch64.whl"
```

Then apply the whole overlay, one file against the pristine wheel:

```bash
cd $SP && patch -p1 < <repo>/tools/main/main1ea7-prod-overlay.diff
find $SP/vllm -name __pycache__ -prune -exec rm -rf {} +
```

[`tools/main/main1ea7-prod-overlay.diff`](tools/main/main1ea7-prod-overlay.diff) is the exact
difference between prod's venv and the pristine wheel: 17 modified files and 6 new ones. Everything
that is not always-on is gated by an environment variable:

| part | files | switch | prod |
|---|---|---|---|
| checkpoint-mapped PLE ([#58439](https://github.com/vllm-project/vllm/pull/58439)) | `config/engram.py`, `qwen4_exp/nvidia/{model_state,ngram_embedding}.py`, `ple_pageable.py` (new) | `--engram-config {"checkpoint_mapped":true}` | on |
| RecoverSSM for GDN ([tools/rssm](tools/rssm/README.md)) | `config/vllm.py`, `mamba/abstract.py`, `qwen_gdn_linear_attn.py`, `qwen4_exp/nvidia/{model,ple_layer}.py`, `recoverssm_gdn.py`, `gdn_recoverssm.py`, `ple_recoverssm.py` (new) | `FN_GDN_RECOVERSSM=1` | on |
| deterministic QSA top-k | `qwen4_exp/nvidia/ops/qsa_indexer.py` + `_C_det.so` ([patches/kernel-det](patches/kernel-det/README.md), `build_det.py`) | `VLLM_QSA_DET_TOPK=1`, `VLLM_QSA_DET_LIB=<path>/_C_det.so` | on |
| bit-stable MoE finalize | `fused_moe/experts/flashinfer_cutlass_moe.py` | `VLLM_MOE_DET_FINALIZE=1` | on |
| FP8 `lm_head` loading (target and MTP), scale naming | `vocab_parallel_embedding.py`, `weight_utils.py`, `qwen4_exp/nvidia/{model,mtp}.py` | always | on |
| 32k draft vocabulary | `v1/worker/gpu/spec_decode/speculator.py`, `v1/spec_decode/llm_base_proposer.py` | `FN_DRAFT_VOCAB=<file>` + `use_local_argmax_reduction` | on |
| NVFP4 draft-head slice | `qwen4_exp/nvidia/mtp.py`, `fn_nvfp4_head.py` (new) | `FN_DRAFT_HEAD_NVFP4=1`, `FN_NVFP4_CFG=64,4` | on |
| experiments, all measured null or not promoted | `linear.py` + `fn_bf16sk.py`, `ple_pageable.py` extras, `weight_utils.py`, `fused_sigmoid_gating.py` | `FN_BF16SK`, `FN_PLE_SYNCTOUCH`, `FN_PLE_POPULATE`, `FN_PFTIME`, `FN_LOAD_DROPCACHE`, `FN_GDN_STORE_CS` | off |

**Any `pip install` of vLLM into this venv silently reverts the whole overlay.** To see what is actually
applied, diff against the unpacked wheel rather than trusting memory:

```bash
LC_ALL=C diff -rq --exclude=__pycache__ --exclude='*.orig*' --exclude='*.pre*' <unpacked-wheel>/vllm $SP/vllm
```

## 4. Serve

[`scripts/serve-flashnext.sh`](scripts/serve-flashnext.sh) is the launcher prod runs. Prod sets these
through a systemd unit with drop-ins:

```ini
Environment=FN_VENV=/opt/llm/runtime/vllm-venv-main1ea7
Environment=FN_MODEL=/opt/llm/models/qwen38-flash-next-mtpfp4
Environment=FN_MAXLEN=32768
Environment=FN_SEQS=16
Environment=FN_SPEC_METHOD=mtp
Environment=FN_SPEC_N=3
Environment=FN_SPEC_NODROP=1
Environment=FN_SPEC_LOCALARGMAX=1
Environment=FN_DRAFT_VOCAB=<repo>/tools/draft_vocab/draft_vocab_32768.txt
Environment=FN_PLE_OFFLOAD=0
Environment="FN_EXTRA=--engram-config {\"checkpoint_mapped\":true}"
Environment=FN_DRAFT_HEAD_NVFP4=1
Environment=FN_NVFP4_CFG=64,4
Environment=FN_GDN_RECOVERSSM=1
```

The flags that are not obvious:

| setting | why |
|---|---|
| `--engram-config {"checkpoint_mapped":true}`, `FN_PLE_OFFLOAD=0` | reads the 47.7 GiB n-gram table in place from the checkpoint's page cache: no table-sized allocation, nothing pinned, no offload worker |
| `--speculative-config` MTP, `num_speculative_tokens` 3, `disable_eagle_block_drop` | the agent loop is −19 % with MTP in this shape; without the block-drop flag, speculation costs a whole prefix block per turn. `k=5` hard-fails (QSA ring capacity; [vllm#54912](https://github.com/vllm-project/vllm/pull/54912)) |
| `FN_GDN_RECOVERSSM=1` | GDN verify from one checkpoint with a per-token replay record, committed once after sampling: −2.4…−4.0 % per cycle at c=1, −5.2 % at c=4, −16 % per agent turn, +37 % KV. PIECEWISE CUDA graphs only (the launcher's default), V2 model runner |
| `VLLM_USE_DEEP_GEMM=0` | DeepGEMM gates on capability *family* 120, which sm_121 satisfies, then faults with `unspecified launch failure` (vllm#54125) |
| `VLLM_GDN_DECODE_KERNEL=triton` | the default CUDA kernel deterministically hangs the engine at c≈32 with FP8 GDN projections. No error; requests just stall |
| `CUTE_DSL_ARCH=sm_121a` | required for the FlashInfer CuteDSL path |
| `--max-model-len 32768` | **not 8192.** One code task emitted 31,115 characters of *thinking* before 12,931 of content |
| `--max-num-seqs 16` | **not 2.** An early "concurrency ceiling" here was this flag, not the hardware |
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | without these, every request carrying `tools` returns **HTTP 400** |

**KV size is a trade against PLE residency.** At the default `--gpu-memory-utilization 0.90` the KV pool
takes ~33 GiB. That leaves ~10 GiB of page cache for the 47.7 GiB table, so a fresh server faults ~30
table pages per decode step, about 11 % slower until it warms ([speed of light §4e](notes/speed-of-light.md)).
[vllm#58835](https://github.com/vllm-project/vllm/pull/58835) shortens that window by 2.35–2.91 ms/step.
All our A/B numbers are measured with `--kv-cache-memory-bytes 4294967296`, which holds ~104k tokens
with RecoverSSM. That is plenty for a few concurrent streams and keeps the table resident; size it for
your own concurrency.

## 5. Verify — capabilities and paths first, then speed

**A serving config has capabilities, not just throughput**, and no speed test sees them:

```bash
# 1. tool calling -- must be 200, not 400
curl -s -o /dev/null -w '%{http_code}\n' localhost:8092/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{"model":"flashnext","messages":[{"role":"user","content":"hi"}],
      "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}]}'

# 2. a generation long enough to clear the thinking block
#    ignore_eos + assert completion_tokens > 0 -- max_tokens is a CEILING, not a target
```

**Then prove each patch is on the path from the log.** A gated patch that silently did not engage
looks like noise:

```text
Mapped PLE table of layer 1 in place: 320001536 rows x 160 B from 10 files; no table-sized device or pinned allocation
FNDV draft vocab: 32768 of 248320 rows (13.2%); draft head 606 -> 45 MiB per draft step (NVFP4 slice, FNNVFP4)
Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128).
Mamba cache mode is set to 'align' for Qwen4ExpForConditionalGeneration by default when prefix caching is enabled
FNRSSM: GDN RecoverSSM speculative verify active (spec_query_len 4)
GDN RecoverSSM path taken: <n> spec rows, align=True
FNRSSM PLE RecoverSSM path taken: <n> spec rows, align=True
```

For the deterministic top-k, check the process, not the log: `grep -c _C_det.so /proc/<VLLM::Worker pid>/maps`
must be > 0.

> ⚠️ **An all-empty cell is not a comparison.** Twice here a determinism check reported five
> outputs "identical" when every one was the empty string: the model was still inside `<think>` and
> the budget ran out. Print character counts and refuse the verdict when the cell is empty.

Then speed. Expected, warm, KV 4 GiB, short prompts ([speed of light §4t](notes/speed-of-light.md)):

| | |
|---|---|
| single stream | **21.49 ms/tok = 46.5 tok/s**, 2.53 tokens accepted per verify cycle |
| 4 streams | 99.7–100.0 tok/s aggregate |
| agent loop, 8 dependent turns | 1.10 s/turn |

Measure with at least two server starts per arm and a warm pass: the first ~minute after a start is
the PLE cold window, and it moves the number by ~11 %.

---

## Things that look like levers and are not

Each measured null here, with the mechanism understood. Details in [closed levers](notes/closed-levers.md)
and [speed of light](notes/speed-of-light.md).

- **Hyper-connection quantization or kernels.** They are latency-bound at ~78 % of roofline: a
  quarter of decode because there are ~102,000 of them, not because any one is expensive.
- **NVFP4 KV cache.** Two independent GB10 measurements plus a structural MTP-acceptance penalty.
  It fails silently.
- **A faster small-M BF16 kernel, streaming stores for the GDN snapshots, the RecoverSSM verify's
  launch config.** All null (§3c, §4l, §4p, §4v).
- **An fp16 SSM state cache.** −2.2 % / −4.4 % per cycle, but a per-token logprob perturbation a fifth
  the size of the whole 4-bit step. A quality trade, not free speed (§4q–4r).
- **Waiting for the host to fault the PLE pages in.** Readahead without any wait is faster (§4x).
- **A "SM121 QSA kernel-guard fix" that widens the sm100→sm120 gate by family.** It circulates in
  at least one popular DGX Spark repo and was **retracted upstream** in sglang#36806: it routes to a
  path that corrupts output at long context (1/4 runs wrong at 120K tokens, 4/4 at 210K, HTTP 200
  throughout). Do not adopt it.

## Known-broken here, independent of anything you do

- **Greedy output is not batch-invariant under concurrency.** Sequential requests reproduce exactly,
  across restarts, with the deterministic top-k and MoE finalize above. Concurrent batches can change
  the text ([determinism investigation](notes/determinism-investigation.md)).
- **Long-prompt hangs (>8k)** were traced to `is_arch_support_pdl()` returning True for anything with
  `major >= 9`, so PDL is used in `_build_qsa_metadata_kernel`, where the dependent kernel waits
  forever (vllm#53960). Not re-checked on `1ea7c63f4`: if a long prompt stalls with no error, check
  this first.
