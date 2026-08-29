# Log

Newest first. Each entry records what was tried, what happened, and what it ruled out.

## 2026-08-27 — the overlay is abandoned; the target is now a specific deadlock

**Weights complete** (126 GiB, 206/206 safetensors).

**The Python overlay approach is dead.** PR #53899 (which is a strict superset of #53896 —
every file, plus the five `vllm/v1/ple_offload/` modules) has a branch that is inconsistent
with itself: `config/model.py` imports `checkpoint_has_lm_head`, which exists on current
`main` but **not** in that branch's own `transformers_utils/config.py`. GitHub reports the PR
`CONFLICTING`, and reconstructing it would mean guessing which main commit each of 85 files
expects. Not worth it when a built image exists.

**Switched to `vllm/vllm-openai:qwen38-flash-next`.** Verified identical to what both reporters
on vllm#53960 used:

    digest  sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
    vllm    0.1.dev20073+g8e685d198

That matters more than convenience: a result on the same build cannot be waved away as a
different-build artefact. It does mean running a container on a box whose serving stack is
otherwise bare-metal — scoped to this experiment; production is untouched.

**The experiment.** `TP=1`, `VLLM_PLE_CPU_OFFLOAD=1`, **`--enforce-eager`**. The two py-spy
dumps on #53960 show the offload worker idle on an empty queue while the main thread spins in
CUDA-graph `replay`, which reads as a blocking host round-trip captured inside a graph. If that
is the mechanism, disabling graph capture avoids it.

- **Serves** → the fault is graph capture of the PLE path, not the offload mechanism, and there
  is a workaround available today for everyone blocked on it.
- **Still hangs** → the offload mechanism itself is at fault and `--enforce-eager` is eliminated.

Both outcomes get reported to #53960. Nobody has publicly tried this yet (checked before
starting, per the rule this project now follows: check the field before anything expensive).

## 2026-08-26 (later) — someone got there first, by another road

[0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
published a working llama.cpp setup the same day. It validates the mmap hypothesis reasoned
about here — the n-gram table is a lookup, never a GEMM, so it can live on NVMe behind the
page cache — and llama.cpp has the primitive to express that directly:

    -ot "per_layer_token_embd=CPU"  -lm mmap

103.7 GiB GGUF (UD-Q4_K_XL), ~76.9 GiB resident, process RSS ~1.4 GiB, ~22 tok/s decode,
~13% degradation from 226 to 19,197 prompt tokens. Load ~3m35s from NVMe.

Three of their findings matter regardless of route:

- **Concurrency crashes**; they run `--parallel 1`. Single-stream only.
- **Speculative decoding bought nothing** — accepted length 2.88, decode stayed ~23 tok/s.
  Their reading: speculation amortises one weight read over k tokens, which does not help
  when the bottleneck is paged embedding lookups.
- **KV is cheap** — 262k context costs ~6 GiB (2 KV heads, mostly linear attention).

**What this does to the case for continuing here.** The headline question — "does it run on a
Spark" — is answered, and the answer is yes at ~22 tok/s, which is *slower than the 27B this
box already serves* (29.5 / 61.6 tok/s on prose / code with DFlash2 at concurrency 16). So
this is not a production upgrade for anyone, and it was never going to be.

What is left open is the part their route cannot reach: concurrency, prefix caching, and a
serving stack that does more than one request at a time. That is the only reason to keep
going with vLLM, and it is a thin reason — worth stating plainly rather than dressing up.

## 2026-08-26 — setup, and the source-build scare

- Model released ~12:29 UTC. `Qwen4ExpForConditionalGeneration`, no vLLM support at
  release; PR #53896 appeared the same day (111 files, unreviewed).
- Sized the builds. NVFP4 (`RadixArk`) at 125.9 GiB is the only one within reach of
  128 GB unified, and still ~9 GiB over what is usable.
- Found the split: main model 78.2 GiB, PLE/n-gram table 47.7 GiB in ten
  `model-plefp8-*` files. Main model alone fits with ~39 GiB to spare.
- Thought a source build was mandatory (`output_gate_type = sigmoid` vs a prebuilt
  kernel defaulting to `silu`). It is not — see
  [why-no-source-build](why-no-source-build.md). This was the decisive finding of the day.
- Prepared the venv (nightly `dev1244+g8d301f075`, chosen because it is 7 commits ahead
  of the PR base and 0 behind, so all 40 modified files match byte-for-byte).
- Staged the 73-file overlay: 0 empty, 0 syntax errors.
- Weights downloading (46% at time of writing).

**Not yet known:** whether the architecture loads at all on sm_121, and whether the
residency gap can be closed. Nothing has been booted.

---

## 2026-08-27 — Root cause: two corrupt shards in my own download

The garbage was never the model, the quantization, the recipe or the environment. Two of the
206 safetensors files are **size-correct and byte-corrupt**:

```
model-bf16-00011.safetensors     dense BF16 body weights
model-plefp8-00000.safetensors   PLE shards 0-12
```

Verified against HuggingFace's published `lfs.sha256`: **204 of 206 clean, these 2 bad.**

### How it happened

The download stalled with two files in flight. I checked the file **sizes** against the HF API,
saw all 418 entries agree, deleted aria2's `.aria2` control files — destroying the resume state
that would have caught it — and recorded the download as verified. The hashes were in the same
API response I was already parsing.

### Why it cost a full day

The failure mode is close to perfectly disguised:

- **It loads.** No error, sane shapes, sane dtypes, correct-magnitude activations.
- **The output is fluent.** Token salad, not NaNs or repetition — it reads like a sampling or
  template problem.
- **It is invariant to configuration**, because it is a property of the weights. Every
  elimination I ran came back clean, and each clean elimination made the wrong conclusion look
  better supported. That invariance is what made "environmental" the obvious inference.
- **Corruption in *both* the body and the PLE** defeated my one good control. The body-vs-PLE
  split was the right experiment; it could not separate them because both were damaged.
- **It survives naive content checks.** I validated the PLE against the official BF16 table at
  cosine 0.999635 — sampling row 0, which sat in the intact head of the corrupt file.

### The rule

Verify `lfs.sha256`, not size. Size agreement between a local file and the HF API says only that
the right number of bytes were allocated — and aria2 preallocates, so a file reaches full size
the moment it starts.

```bash
curl -s "https://huggingface.co/api/models/$REPO/tree/main?recursive=true&blobs=true" \
  | jq -r '.[]|select(.lfs)|"\(.lfs.sha256)  \(.path)"' > SHA256SUMS
sha256sum -c SHA256SUMS
```

### Retracted

- [blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) — I asked them a driver /
  firmware question about a fault that was mine. Their recipe and mmap hook are fine.
- [dolf3131#1](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark/issues/1) — I asked for a
  `*.self_attn.*` exclusion control on a premise that no longer stands. Their `group_size`
  elimination and their parameter accounting for the 3.3 GiB RadixArk/Inferact gap are
  independent of my failure and still hold.

Re-fetching the two files. Next entry reports whether RadixArk then serves coherently under the
one-gate patch — which is the part of the earlier report that still matters, since it would turn
the published "loads but emits garbage" table line into "works with a one-line change".

## 2026-08-29 — a day of nulls, and the measurement that explained them

Chased the largest item in the decode profile — **25–27% of GPU time in
`cutlass_80_wmma_tensorop_bf16`**, the BF16 hyper-connection GEMMs, Ampere-generation kernels on a
Blackwell part. Three interventions, all null against a **measured 6.9% noise floor** (six
identical runs: 34.7–37.1 tok/s, mean 35.68, sd 0.84):

| attempt | what changed | result |
| --- | --- | --- |
| blockwise FP8 (`FP8_PB_WO`) | precision | slower |
| CUTE-DSL skinny GEMM, gate + M=3 entry | kernel, fused down projection | 35.92 vs 36.45 |
| per-channel FP8, 96 `_up` projections | half the bytes **and** the faster kernel | 36.05 vs 36.45 |

### Four layers of "installed but not running"

Each presented identically — a null inside the noise floor — and each was only visible after
clearing the one above it:

1. **arch gate** — `enable_qwen38next_low_latency_gemm` returns early unless `_is_sm103()`
2. **plan table keyed by exact M** — `plan.get(x.shape[0])`, a miss falls to `F.linear` silently
3. **cudagraph replay** — the custom op's Python body runs only at capture, so a call-count log
   never fires, and the M that matters is the **capture size**, not the token count
4. **a stale compiled graph** — three six-run arms agreed to 0.2% because they were *one cached
   graph*, not three arms

Stopping at any of the first three would have published "the skinny GEMM does not help on GB10",
confidently and wrongly.

### The explanation, which took two minutes and should have come first

Per-shape microbenchmark (`tools/shapebench.py`), L2 defeated by rotating ~300 MB of weights,
every timing printed against its roofline:

| shape | M=1 BF16 | M=1 FP8 | speedup | roofline |
| --- | --- | --- | --- | --- |
| (10240, 320) hyper up | 30.8 us | 26.1 us | 1.18x | 24.0 us |
| (336, 10240) hyper down | 37.0 us | 41.5 us | **0.89x** | 25.2 us |
| (10240, 2560) GDN in_proj | 307 us | 126.7 us | 2.42x | 192 us |

These GEMMs run at **~78% of roofline already** — latency-bound, not bandwidth-bound. They are a
quarter of decode time because there are **~102,000 of them** (~200 per forward), not because any
one is expensive. FP8 saves ~4 us/call on `up` and *loses* 6–9 us/call on `down`, so the halves
cancel: predicted ~1.3%, measured −1.1% ± 3.0%. **The nulls were the correct answer.**

### Two things I published and withdrew the same day

- **"The profile was prefill-contaminated."** Plausible — the trace was 16k prefill vs 2k decode
  tokens — but a decode-only re-profile gave the same ranking (fp8 34.5 vs 32.2, wmma 26.6 vs
  25.0, MoE 22.2 vs 22.1). Refuted by testing it.
- **"The torch.compile cache key omits kernel selection."** Checked before filing:
  `KernelConfig.compute_hash()` *is* wired in at `vllm/config/vllm.py:506`. What is not hashed is
  an edit to vLLM's own source, which is expected. Would have been a wrong report.

### Upstream

- **[vllm#54367](https://github.com/vllm-project/vllm/pull/54367)** — opened against `main`:
  `FP8_PER_CHANNEL_PER_TOKEN` is defined but has no branch in the `MIXED_PRECISION` dispatch, so
  the layer falls through to unquantized and the checkpoint fails to load. Fix + unit test.
- **[branch `gb10-sm121-fixes`](https://github.com/jschmied/vllm/tree/gb10-sm121-fixes)** — three
  commits on #53896's head; commented there about `GatedResidual` hardcoding `quant_config=None`
  while its own docstring claims dispatch works.
- **[0xBakeer#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6)** landed and closed
  with attribution; our noise floor and MTP-off A/B are in his docs, credited.
- Answered `Chinmayrawat15` on #54097 — his `Exception`-catching generalisation is better than the
  `OSError` enumeration I proposed, and I declined the follow-up he offered.

### The rule

**Verify the lever is real at the shape level before building anything.** Ranking by profile is
necessary and not sufficient: a layer can be a quarter of GPU time and still have nothing to give,
because share of time and headroom are different quantities. The microbenchmark that settled this
costs two minutes with the server stopped, and would have pre-empted a checkpoint build, four
failed server starts and three six-run A/B arms.

## 2026-08-29 (evening) — two capability defects the whole benchmark suite was blind to

Everything above this line measured speed. The evening found two things that decide whether the
model is *usable*, and neither was visible to any speed test.

### Tool calling was rejected outright

`serve-fnext.sh` set `--reasoning-parser qwen3` and nothing else. Every request carrying `tools`
returned **HTTP 400**:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

An omission, not a decision — both Laguna launchers already set theirs. Fixed with
`--enable-auto-tool-choice --tool-call-parser qwen3_xml`; now **32/32** across temps 0.2 / 0.6 /
1.0 / default, correct function name every time. Full write-up in `tool-calling-was-off.md`.

### The context window could not hold the model's own reasoning

`--max-model-len 8192`, chosen for throughput benchmarking, rejected a code task asking for 26k
output tokens. Raised to 32768, at which point the same task ran: **31,115 characters of reasoning
before 12,931 characters of content.** 8192 was never going to serve real work.

**Both were invisible to throughput, acceptance, NLL, divergence and coherence tests, because none
of those sends a `tools` field or a long generation.** A serving config has capabilities, not just
speed.

### Task A (Go): FAIL, genuinely

At vendor sampling (1.0 / 0.95 / 20, from `generation_config.json`) with
`reasoning_effort=medium`. Both blocks extracted cleanly, file complete, zero stray fences — and
it does not compile:

```go
func isExpired(e *entry[_, _]) bool     // cannot use _ as value or type
```

`_` is not a valid type argument; the helper needed to be generic. One line fails the build while
the cache implementation and a 313-line test file are structurally sound. **N=1 at temperature
1.0** — provisional, and our own scorecard has two rows that were wrong until re-run.

Three settings had to be right first, each of which would otherwise have produced a fake failure:
`:8092` not `:8080`; vendor sampling rather than the generic `0.7/0.95/40` fallback; and
`reasoning_effort=medium`, since the template defaults to `xhigh` which spends the whole budget in
`<think>`.

### lm_head under speculation, measured properly

The +11% was measured with speculation **off** while production runs MTP k=2, and the drafter has
no head of its own (`mtp.shared_head.head.` → `lm_head.`). Matched A/B:

| | BF16 head | FP8 head |
| --- | --- | --- |
| decode tok/s | 30.60 | **36.45** |
| acceptance | **60.3%** | 56.6% |

The FP8 head **does** cost draft quality (−3.7 pp) and wins **+19.1%** anyway. The gain nearly
doubles versus speculation-off, confirming the compounding argument by measurement. Published
"+11%" needs qualifying as speculation-off.

## 2026-08-30 — `--max-num-seqs` 16 → 64: no effect (negative result)

0xBakeer published a SEQS A/B showing 1.2–2.7× on loaded rows, and every concurrency number we had
was taken at **c=16 against a cap of exactly 16** — so the cap bound precisely where we measured,
and our flat ~100 tok/s aggregate was suspect as an artifact rather than a ceiling. It isn't.

Only `max_num_seqs` changed; same checkpoint (FP8 head), MTP k=2, 8192 ctx, 4000-token inputs.

| | SEQS=16 | SEQS=64 |
|---|---|---|
| c=1 decode tok/s | 36.45 ± 1.04 | 36.83 (+1.1%) |
| c=16 aggregate | 99.1 / 101.5 / 100.1 | 97.2 / 101.4 / 100.7 (−0.3%) |
| c=32 aggregate | not reachable | 110.1, TTFT **30.7 s** |

Both moves are inside the 6.9% noise floor. **~100 tok/s at c=16 is a real bandwidth ceiling**, and
c=32 buys +10% aggregate for 4× the TTFT — not a trade worth making for agent work, where we rank
by TTFT. 0xBakeer's result does not transfer; the likely reason is that their baseline cap was low
enough to bind (ours was already at the knee).

**Left open by this:** an old uncapped run recorded 266.8 tok/s at c=48, far above the 110 here. The
configurations differ in more than concurrency — most importantly MTP was off — and aggregate swings
2.6× with prompt length, so the two are **not comparable** and no conclusion is drawn. The real
question it raises is worth a clean test: **does MTP k=2 cost aggregate throughput under load?**
Speculation spends compute on rejected drafts, which is cheap at c=1 and expensive when the batch is
already saturating bandwidth. One A/B at c=16 and c=32, MTP on vs off, same inputs.
