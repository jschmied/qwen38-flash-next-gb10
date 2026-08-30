# Open work, ranked

Written 2026-08-30. Each item says what it is, what it is worth, and what would settle it.
Anything measured-and-closed lives in its own note, not here.

## Blocked on someone else

- **Push `gb10-sm121-fixes` / open its PR.** Branch is rebased onto vllm#53896 head `91a6b555`,
  three commits, pushed to <https://github.com/jschmied/vllm/tree/gb10-sm121-fixes>. Its code
  exists only on the PR branch, so it cannot target `main`; the route is the comment already on
  #53896. Nothing to do until that PR moves.
- **SWE-bench Multilingual-28 (Go/Java).** Needs the x86 box `10.0.0.8` — currently *no route to
  host*. arm64 has **no** Java/JS eval images, so there is no fallback. When it returns: tunnel
  must cross ports (`-R 8080:127.0.0.1:8092`), and `--model openai/flashnext`.

## Closed since this file was written (2026-08-30)

- **`--max-num-seqs` 16 → 64** — null (−0.3% at c=16, +1.1% at c=1, both inside the 6.9% noise
  floor). ~100 tok/s at c=16 is a real bandwidth ceiling, independently corroborated by 0xBakeer's
  vLLM recipe landing at 96–109. Their 1.2–2.7× came from a baseline of **2** slots, not 16.
- **MoE backend axis** — `moe_backend='auto'` already resolves to `FLASHINFER_CUTLASS` here and
  always has, so it was never an untried lever; `b12x` is rejected; `triton`/`cutlass` hit the
  SM120/121 SMEM overflow. The five remaining backends have no field evidence favouring them.
- **vllm#53960 PLE deadlock** — fixed upstream by `4e8b849b8d97`, backported into our venv,
  verified serving at no cost. We could not reproduce the hang and said so upstream.
- **SGLang as a stack switch** — blocked: sgl#36558 reports Flash-Next unservable on SM121, and
  both the base support (#36497) and the resolver fix (#36556) are unmerged. `is_sm120_supported()`
  gates on **major 12**, so #36556 does cover GB10 — the blocker is that it has not landed, and it
  carries no GB10 verification.

## Worth doing, in order

1. **Repeat Task A (Go), n≥4.** Current result is FAIL at N=1 on one generics error
   (`entry[_, _]`), at temperature 1.0. Our own scorecard has two rows that were wrong until
   re-run. A 2/4 or 3/4 is worth far more than one FAIL. Then Task B (Java) for the pair.
2. **FP8 QSA KV** (from `alesha-pro/qwen38-flash-next-4x3090`). Roughly doubles the KV pool, which
   now matters at 32k context. Needs: QSA dtype/scale plumbing, an FP8 decode kernel, and
   **calibrated scales — no scale=1 fallback**. Not upstream (deferred post-merge on #53896), so
   it is a fork decision. Verify all 12 QSA layers *log* their scales.
3. **DFlash2 re-measure.** vllm#52816 merged 2026-08-21 (`b389ac29`), which unblocks the queued
   comparison. Use the merged main, not the abandoned 0.27.1 port (that one loses 5.7% decode and
   0.24 accept length).
4. **INT8 lm_head** (from `styles01`, `patch_int8_lmhead_v3.py`). 3.35 ms vs 8.8 ms at B=1,
   argmax-exact, frees ~1.4 GiB. Ours is FP8 blockwise at +19.1% under MTP. Whether INT8 beats it
   is unmeasured; the patch targets vLLM 0.25.1 paths so porting is real work.
5. **Acceptance gap.** Ours is 56.6% at k=2 where YSLAB report 73.7% on the same architecture.
   Not explained by the FP8 head (BF16 head only reaches 60.3%). Most likely workload — our bench
   continues random real-corpus slices, deliberately unpredictable. Settle with an easier prompt
   set; acceptance is the one lever that scales throughput without touching bytes or kernels.

## Cheap, do alongside anything

- **Add to the offline gate**: shared-expert *gate* must stay BF16 (alesha-pro's rule; we comply by
  inheritance, not by check).
- **Add to the bench harness**: accept-length pinned at maximum is a **corruption signature**, not
  health — one field case read 3.00/3 while GSM8K scored 0/10. We already read the counters.
- **Carry the KLD caveat**: our divergence harness must not label top-N KLD as full-vocabulary KLD.

## Do not spend time on

- **Hyper-connection quantization or kernels** — three interventions, all null, mechanism
  understood (latency-bound at ~78% of roofline; ~102k calls). Corroborated three ways
  independently. See `why-the-hyper-connections-do-not-respond.md`.
- **NVFP4 KV** — closed by two independent GB10 measurements plus a structural MTP-acceptance
  penalty, and it fails silently.
- **FlashInfer AOT prebake** — `flashinfer-jit-cache` already ships 960 prebuilt `.so`; we invoke
  ninja zero times. Keep it as a post-driver-upgrade recovery procedure only. **Do not bump
  flashinfer past 0.6.17** — 0.6.18 drops SM121a cubins from the aarch64 cu130 wheel.
- **Lowering `gpu-memory-utilization` to avoid freezes** — refuted; 0.70 is the worst recorded
  outcome. The cause is absolute free memory at launch, not the ratio.
- **The sm_121 gate on the CUTE-DSL skinny GEMM** — measured null with stock configs
  (36.45 → 35.92). dolf3131 got +6.9% only with hand-swept TP=1 configs.

## Standing rules earned the hard way

- **Profile the regime you will measure**, and **verify the lever is real at the shape level before
  building anything** (`tools/shapebench.py`, two minutes).
- **A serving config has capabilities, not just speed.** Probe tool calls and a long generation
  before benchmarking a new recipe. Both of tonight's defects were invisible to every speed test.
- **Prove a kernel ran** — log first-sight dispatch keys inside the op. A call-count threshold
  never fires under cudagraph replay.
- **Clear `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`** when benchmarking a source-level patch;
  a stale graph replays unpatched code. (Config flags *are* hashed correctly — checked.)
- **Noise floor is 6.9%** (n=6). Nothing under ~10% is callable from single runs.

- **MTP on/off under load.** Does k=2 cost aggregate throughput at c=16/c=32? Speculation is nearly
  free at c=1 and should get worse as the batch saturates bandwidth. Same inputs, one A/B. Raised by
  the SEQS negative result (`log.md`, 2026-08-30) — do **not** compare against the old 266.8 tok/s
  c=48 row, which differs in more than concurrency.

- **FP8 KV — re-scoped, and check the dtype gate first.** sgl#36797 measures fp8_e4m3 KV as
  ~speed-neutral against bf16 on SM121, so the win is **context/concurrency headroom, not tok/s**.
  sgl#36545 shows fp8 KV + QSA asserting on **BF16 q vs FP8 k** in another stack. Read vLLM's QSA
  decode dtype handling before any build — ten minutes against a multi-hour detour.
- **SGLang — watch, do not attempt.** sgl#36558 reports Flash-Next *unservable* on SM121 (QSA
  decode resolver gates on `is_sm100_supported()`); base support sgl#36497 and the resolver fix
  sgl#36556 are both unmerged. Revisit when #36556 lands.
