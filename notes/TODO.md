# Open work, ranked

Rewritten 2026-08-31. Anything measured-and-closed lives in its own note; this file says only what
is open, what is blocked, and what is settled enough not to revisit.

## Open, in order

1. **Corroborate or refute vllm#54521's `indexer_budget` model.** They report greedy decoding
   deterministic *below* `indexer_budget` (2048) and non-deterministic above, because QSA switches
   to top-k selection — which would explain our own unexplained non-determinism. **Our first run
   contradicts them:** divergence at 582 and 1,142 prompt tokens, both well below the budget. But we
   run MTP k=2 and their repro does not, so the decisive cell is MTP-off below the budget. In
   flight. Either outcome is worth reporting; the issue has no comments.
2. **Quality scorecard: repeat Task A (Go), n≥4, then Task B (Java).** Current result is a single
   FAIL at temperature 1.0 on one generics error (`entry[_, _]`). Two rows of our scorecard were
   wrong until re-run; a 2/4 or 3/4 is worth far more than one FAIL.
3. **DFlash2 re-measure.** vllm#52816 merged 2026-08-21. Use merged main, **not** the abandoned
   0.27.1 port (−5.7% decode, −0.24 accept length).
4. **Acceptance gap** — ours 56.6% at k=2 against 73.7% and 75.6% reported elsewhere. ⚠️ Re-scoped:
   DJLougen measured acceptance collapsing 33.3% → 3.7% on **batch geometry alone**, with the output
   diverging. An acceptance number without its batch geometry may not be comparable at all, so
   settle the geometry before chasing the gap.
5. **MTP under concurrent load** (c=16/c=32). Nearly free at c=1; should worsen as the batch
   saturates bandwidth. Single-stream and agent-loop behaviour are now both measured — this is the
   remaining axis.
6. **INT8 `lm_head`** (styles01's `patch_int8_lmhead_v3.py`) against our FP8 blockwise. 3.35 ms vs
   8.8 ms at B=1 reported, argmax-exact, frees ~1.4 GiB. Their patch targets 0.25.1 paths, so
   porting is real work and the payoff is unmeasured.

**Needs a decision, not a measurement:** the **NVFP4 PLE checkpoint** (`provsalt/…-PLE-NVFP4` plus
the `qwen38_nvfp4_ple` plugin) is ~26.8 GiB against our 47.7 at FP8 — roughly **21 GiB back**. It is
a large download; ask before starting.

## Blocked on someone else

- **`gb10-sm121-fixes`** — ⚠️ **unblocked as of 2026-08-31**: #53896 merged, so these three commits
  can now target `main` instead of a PR branch. They need rebasing onto the post-merge tree, where
  the package is `qwen4_exp`.
- **SWE-bench Multilingual-28 (Go/Java)** — needs the x86 box `10.0.0.8`, currently *no route to
  host*. arm64 has no Java/JS eval images, so there is no local fallback. When it returns: the
  tunnel must cross ports (`-R 8080:127.0.0.1:8092`), and `--model openai/flashnext`.
- **SGLang** — sgl#36558 reports Flash-Next unservable on SM121; base support (#36497) and the
  SM120/121 resolver fix (#36556) are both unmerged. `is_sm120_supported()` gates on **major 12**, so
  #36556 *does* cover GB10 — the blocker is only that it has not landed. Watch, do not attempt.

## Settled — do not re-open without new evidence

- **Hyper-connections** — three interventions, all null, mechanism understood (latency-bound at ~78%
  of roofline across ~102k calls). Corroborated three ways.
- **NVFP4 KV** — three independent GB10 measurements, plus a structural MTP-acceptance penalty, plus
  silent failure. (**FP8** KV is different and now shipped-capable: ×1.72 pool, see `fp8-kv.md`.)
- **MTP k=1** — strictly dominated. Same one-block cache cost as k=2 for +31% decode instead of
  +56%. Never use it.
- **`--max-num-batched-tokens` 8192** — reversed at n=3; the apparent +10–12% was prefill noise.
  Stays at 4096.
- **`--max-num-seqs` 16 → 64** — null. ~100 tok/s at c=16 is a real bandwidth ceiling,
  independently corroborated at 96–109.
- **MoE backend axis** — `auto` already picks `FLASHINFER_CUTLASS`; b12x is selectable once the
  drafter is quantized and then faults (vllm#50189). See `moe-backend-axis.md`.
- **FlashInfer AOT prebake** — the jit-cache wheel already ships 960 `.so`; we invoke ninja zero
  times. ⚠️ **Do not bump flashinfer past 0.6.17** — 0.6.18 drops SM121a cubins from the aarch64
  cu130 wheel.
- **Lowering `gpu-memory-utilization`** — refuted; 0.70 was the worst recorded outcome, and the
  runbook it came from is dual-node Ray+EP where the growth is in pools utilization does not bound.
- **The sm_121 gate on the CUTE-DSL skinny GEMM** — null with stock configs (36.45 → 35.92).

## Cheap, do alongside anything

- **Gate**: shared-expert *gate* must stay BF16 — we comply by inheritance, not by check.
- **Harness**: accept-length pinned at maximum is a **corruption signature**, not health (one field
  case read 3.00/3 while GSM8K scored 0/10).
- **Harness**: detect empty content by **`finish_reason: "length"`**, not by counting characters
  afterwards — that let a determinism probe call five empty strings "identical", twice.
- **Carry the KLD caveat**: top-N KLD is not full-vocabulary KLD.

## Standing rules earned the hard way

- **Noise floor 6.9% — for *decode*.** ⚠️ **Prefill is ±20%** (1,633–2,367 tok/s within one
  configuration). A prefill claim needs n≥3 and a wider bar.
- **Verify the lever at the shape level before building** (`tools/shapebench.py`, two minutes).
- **A serving config has capabilities, not just speed** — probe tool calls and a long generation
  before benchmarking a recipe.
- **Prove a kernel ran**; a call-count threshold never fires under cudagraph replay.
- **Clear `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`** for source-level patches.
- **A gate must ask the consumer's question** — resolve the runtime's name, read as the serving uid,
  parse the file the runtime parses. Ours passed three times while the server failed.
- **When two config edits fail identically, stop editing and instrument.**
- **`max_tokens` is a ceiling, not a target.** Use `ignore_eos` to force a generation length.

## The MTP depth sweep was cut short by a misread constraint (2026-08-31)

`k=5` hard-fails with `QSA ring capacity 12 must divide the attention block size 848`, and we
recorded that as an upper bound. It is not one. From `qsa_cache.py:778-783`:

    span     = compress_ratio + n
    capacity = compress_ratio * cdiv(span, compress_ratio)
    assert block_size % capacity == 0

With `compress_ratio = 4` and `block_size = 848` that makes **n = 0..4 and 9..12 legal, n = 5..8
illegal** — a hole, not a ceiling. (At `block_size = 1600`, 13..16 open up as well.)

We swept k = 1, 2, 3 and called k=2 optimal. **k=4 was never tried, and the entire 9..12 band was
never tried.** Since decode here is bytes-per-token-bound and speculation is the only remaining way
to get more tokens per weight read, a deeper band that we wrongly believed illegal is the most
concrete untested speed lever we have.

Caveat before spending a night on it: acceptance falls with depth, and on other models in this
fleet the curve is an inverted U that turns over by n≈3-4. The 9..12 band is worth **one** arm to
see whether acceptance has collapsed, not a full sweep on spec.
