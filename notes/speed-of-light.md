# Speed of light for Flash-Next decode on GB10 (step 1 of the waste audit, 2026-09-24)

**Question.** How far is c=1 decode from the floor set by the bytes it must move, and where is the rest?

**Method.** `tools/sol/headers.py` and `tools/sol/floor.py` (run from a directory where `headers.py` has written
`tensors.json`); output in `notes/data/sol-floor.txt`.
- Every tensor's dtype and size comes from the mtpfp4 safetensors headers; state and KV sizes come from
  config.json.
- One MTP n=3 verify cycle at c=1 reads:
  - the target once over 4 tokens: FP8 dense, the BF16 leftovers, the FP8 lm_head, and the routed experts
    actually touched;
  - GDN fp32 state, read plus written;
  - QSA selected K/V (≤ 2048 tokens) plus the compressed indexer keys;
  - the drafter 3×: MTP dense in BF16, the 45 MiB NVFP4 draft head, and experts, 10 per decode step.
- Floor = bytes / 220 GB/s, our measured bandwidth (det-231). At the 273 GB/s spec sheet every floor drops ~20 %.

**Sizes:**

| part | bytes per cycle |
|---|---|
| target FP8 dense | 2,551 MiB |
| target BF16 leftovers (+ norms, scales) | 1,912 MiB, of which hyper-connection mixers 1,209, shared experts 450, router 120 |
| lm_head (FP8) | 606 MiB |
| routed experts | E × 48 × 2.637 MiB (E = distinct experts per layer across the 4-token verify batch) |
| GDN state | 108 MiB read; 108–432 MiB written (1 or 4 speculative positions) |
| QSA | ~50 MiB at any context (selection-capped) |
| drafter, per cycle | 739–828 MiB (MTP dense BF16 173 MiB × 3, draft head 45 × 3, experts) |

**Floor vs measured.** Measured: 23.36 ms/tok × 2.535 = **59.2 ms per verify cycle** (finding 234, NVFP4 draft head,
c=1).

| E (distinct experts, 4 tokens) | floor ms/cycle | floor ms/tok | measured ÷ floor |
|---|---|---|---|
| 10 (fully shared routing) | 35–37 | 13.8–14.5 | 1.6–1.7× |
| 20 | 41–43 | 16.2–16.9 | 1.4× |
| 30 | 47–49 | 18.6–19.3 | 1.2× |
| 39.3 (independent routing) | 53–55 | 20.9–21.6 | 1.1× |

Context length barely moves the floor (1k → 32k adds ~0.3 ms), because QSA reads only its selection.

**Reading:**
- **The decisive unknown is E.** Near-independent routing puts us at ~1.1× the floor, so only fewer bytes help:
  - the BF16 leftovers at 8 bits save ~0.95 GiB/cycle ≈ −8 % (the INT8/MXFP8 lever, now sized);
  - the MTP dense layers at FP8 save ≈ −2 %;
  - writing GDN state for only the accepted position saves ≤ −2.5 %.
- **Correlated routing (E ≈ 10–20) would leave 30–40 % unexplained:** launch gaps, inefficient kernels, redundant
  work. The step-2 kernel audit targets exactly that.
- The routing capture used by det-198 (`/opt/llm/calib/capture`) no longer exists. **Step 2 must log live MoE
  top-k ids** alongside the kernel profile to pin E.

**Step 2 (planned, prod-down, after the FULL-graph A/B):** a torch profile of steady-state c=1 and c=4 decode, plus
logged routing:
- kernels mapped to modules (`kernelroster.py`, `prof_attrib.py`);
- each group's floor from its bytes, sorted into necessary-efficient, necessary-inefficient and unnecessary
  (copies, casts, repeated metadata, rejected-draft verify work, drafter re-prefill, graph padding).

## Plain decode (no speculation): the exact floor

Without MTP each token routes to exactly 10 experts per layer, so nothing is unknown: 6.6 GiB per token →
**31.5 ms/tok at 220 GB/s (31.8 tok/s), 25.4 ms/tok at 273 GB/s (39.4 tok/s).** The last measured no-spec c=1 decode
was 40.2 ms/tok (finding 210, older dev524 stack), i.e. 1.28× the 220 GB/s floor.

**This floor does not bound speculative decoding.** With MTP n=3 one verify cycle reads the target weights once for
4 tokens (26.6 distinct experts per layer, not 40) and yields 2.53 accepted tokens. The floor that applies to prod
is 45.2 ms per cycle ≈ **17.9 ms/tok (~56 tok/s)** — step 2a below. Prod measures 21.49 ms/tok (46.5 tok/s, §4t),
1.2× that floor. The per-token floor scales with acceptance: at 3.0 tokens per cycle it would be ~66 tok/s, minus
the slightly larger reads of deeper drafting.

## Step 2a — live capture + per-module bench (2026-09-24 night, `tools/fncap/`, data `notes/data/fncap/`)

One eager start in the prod config (`12-fncap`). The capture ran on real traffic: c=1 essays, c=4, and the agent
loop.

**Routing, measured** (top-10 ids from every router call, all 48 layers):

| call rows | distinct experts per layer, mean | independent would be |
|---|---|---|
| 4 (c=1 verify batch), n = 69,648 calls | **26.6** (median 27) | 38.8 |
| 8 | 47.3 | 74.7 |
| 16 (c=4 verify) | 82.5 | 138.6 |
| drafter, 1 row | 10.0 | 10.0 |
| drafter, 4 rows (its prefill of the verified tokens) | 35.7 | 38.8 |

**Floor with measured routing: 45.2 ms per verify cycle at 220 GB/s** (36.4 at 273). By part:

| part | ms |
|---|---|
| target dense | 21.3 |
| target experts | 16.1 |
| lm_head | 2.9 |
| drafter (dense, head, experts) | 3.8 |
| GDN state | 1.0 |
| QSA | 0.2 |

Measured 59.2 ms → **1.31× the floor, a gap of 14.0 ms per cycle** (17.8 ms/tok floor vs 23.4 measured).

**Per-module bench** (inside the worker, real inputs at M=4, 64 MiB memset before each call to evict L2, eager
event timing and CUDA-graph timing):

| module | graph µs | floor µs | ratio |
|---|---|---|---|
| FP8 `in_proj_qkvz` (GDN) | 197.5 | 190.7 | **1.04** |
| FP8 `qkv_proj` (QSA layer) | 164.9 | 154.9 | **1.06** |
| FP8 `out_proj` / `o_proj` | 78.9 / 82.2 | 71.5 | 1.10 / 1.15 |
| BF16 shared expert (block) | 55.5–56.4 | 44.7 | 1.24–1.26 |
| BF16 router gate | 9.1–16.6 | 11.9 | 0.76–1.39 (tiny weights partly survive the flush) |
| BF16 `in_proj_ba` (96×2560) | 15.2 | 2.2 | 6.9 (fixed per-launch cost; ~0.5 ms/forward over 36 layers) |

Eager timings are 1.4–1.9× the graph timings, but that is per-call launch latency that a pipelined forward hides
(finding 237: FULL decode graphs are a null).

**Correction to the historical ranking.** det-134 put the small-M FP8 blockwise GEMM at "~2.5× its byte floor, the
largest kernel-level waste". Measured on real inputs with L2 flushed, it is at **1.04–1.15×**. That lever is gone,
and the 14 ms gap is elsewhere. Not yet measured: v1 never reached the MoE experts and hyper-connection mixers
(called with kwargs) or the lm_head (applied through the logits processor). **Step 2b** (`20-fncap2`) benchmarks
those.

## Step 2b — experts, lm_head (`20-fncap2`, `notes/data/fncap/bench2.json`)

| module | graph µs | floor µs (220 GB/s) | ratio |
|---|---|---|---|
| FP8 lm_head [248320×2560], M=4 | 2,729.7 | 2,890 | **0.94** (effective ~234 GB/s: our 220 GB/s is slightly conservative) |
| target MoE block, layer 3, 4 rows | 435.4 | 391 at E=26.6 (routed + shared + router) | **1.11** |
| target MoE block, layer 4, 4 rows | 383.9 | 391 | **0.98** |
| drafter MoE block, 1 row (E=10) | 240.6 | 182 | 1.32 |
| BF16 `in_proj_ba`, M=4 | 15.7 | 2.2 | 7.1 (fixed launch cost, as in 2a) |

**Instrument note.** `MoERunner` takes `hidden_states` and routes internally. Its kwarg named `router_logits` is the
`[4, 2560]` hidden state, so `bench.json`'s `distinct_experts` (28/30) and `floor_us` (71/76) for the MoE blocks are
**wrong**. The table uses the measured average E = 26.6 from the routing log. The `draft.lm_head` entry is the full
head, but drafting uses the NVFP4 slice through `get_top_tokens`, so that entry is ignored.

**So far every large module sits at 0.94–1.15× its byte floor:** FP8 dense, lm_head, and the target MoE. Only the
drafter MoE at M=1 (1.32×) and the tiny `in_proj_ba` run well above their floors, and neither explains a 14 ms gap.
**Still unmeasured:** the hyper-connection mixer linears, which the model calls from `GatedResidual.mix` /
`combine_and_mix`, so the v2 hooks on the parent never fired. `30-fncap3` targets their sub-linears. If they are
near their floor too, the gap is not in any single dense module, and a whole-step kernel profile (GDN, QSA,
elementwise, inter-kernel gaps) is the next instrument.

## Step 2c — hyper-connection mixer linears (`30-fncap3`, `notes/data/fncap/bench3.json`)

| module (M=4 target, M=1 drafter) | graph µs | floor µs | ratio |
|---|---|---|---|
| mixer `input_mix_weight_down_block_inject` [324×10240] BF16, layers 3/4 attn+mlp | 33.6–35.5 | 31.3 | **1.07–1.13** |
| mixer `input_mix_weight_up` [10240×320] BF16, layers 3/4 attn+mlp | 28.9–30.3 | 29.8 | **0.97–1.02** |
| top-level mixer down / up | 32.8 / 30.0 | 29.8 | 1.10 / 1.01 |
| drafter mixer down / up (M=1) | 34.5 / 29.1 | 29.8 | 1.16 / 0.98 |
| drafter `o_proj` BF16 (M=1) | 134.1 | 143.0 | 0.94 |

The older profile's single-warp sm80 WMMA kernels for these shapes (`where-the-gpu-time-goes.md`) are gone on this
stack: **the mixers run at their byte floor.** **[Withdrawn in step 3: the prod PIECEWISE path still uses those kernels; mixer down is 40.5 µs in-model.]**

**Conclusion of step 2: no dense module is meaningfully above its byte floor.** FP8 dense runs at 1.04–1.15×, BF16
mixers/shared/router at 0.97–1.26×, lm_head at 0.94×, target MoE at 0.98–1.11×. The only outliers are small: the
drafter MoE at M=1 (1.32×, ~0.1 ms per draft step) and `in_proj_ba` (7×, but ~0.5 ms per forward in absolute
terms). The 14 ms/cycle gap to the 45.2 ms floor must therefore sit where module benches cannot see:
- GDN recurrence, conv and gating; the QSA indexer, top-k and attention; PLE;
- norms, RoPE, activation quant and elementwise work between modules;
- sampling and rejection;
- inter-kernel idle.

**Step 3** (`40-prof`, hypothesis in `tools/prof/HYPOTHESIS.md`) profiles one whole step in the prod config to
attribute it.

## Step 3 — whole-step kernel profile in the prod config (`40-prof`, 2026-09-25 00:40)

Prod config (PIECEWISE compiled, MTP n=3, NVFP4 draft head), torch profiler over 54 c=1 decode steps, first 5 skipped.
Parsers: `tools/prof/stream.py` (streams the 300 MB trace line by line, ~10 MB RSS), `an2.py` (per-step categories),
`an3.py` (stream overlap). Summary: `notes/data/prof-0925-summary.txt`; profiler table
`notes/data/prof-0925-profiler_out.txt`. Profiled steps took 55.0 ms. The unprofiled chunks at the start of the same
run took 59.6 ms, most likely the ~6 % warm-up drift after a restart, so the profiler did not slow anything down.

**The GPU is not idle:** busy (union of all streams) is 52.7 ms of the 55.0 ms step, so idle is 2.4 ms (4 %). This
agrees with finding 237 (FULL graphs null). The routed MoE runs on per-layer aux streams: 20.7 ms/step, but only
4.6 ms of it overlaps the main stream, so the step is essentially serial.

| category (ms/step) | measured | byte floor (220 GB/s) | ratio |
|---|---|---|---|
| MoE grouped GEMM (`GemmUniversal`, aux streams) | 18.2 | ~16.9 (target E=26.6 + drafter) | 1.08 |
| FP8 blockwise GEMM (dense + lm_head) | 16.3 | 15.1 | 1.08 |
| **BF16 GEMM (cuBLAS `cutlass_80_wmma` 32-thread blocks, 424 calls) + gemv** | **16.7** | ~11.6 (target 9.1 + drafter 2.5) | **1.44** |
| GDN `fused_sigmoid_gating_delta_rule_update` | 1.5 | 1.0 (state read+write) | 1.5 |
| MoE routing/finalize, elementwise, norms, QSA, NVFP4 head, act-quant, other | ~5.3 | — | — |

Target BF16 GEMMs in the model, per call (grid → shape, µs):

| linear | calls/step | in-model | floor | in-model ÷ floor |
|---|---|---|---|---|
| mixer down [324×10240], split-K 9 | 100 | 40.5 | 30.2 | 1.34 |
| mixer up [10240×320] | 100 | 33.9 | 29.8 | 1.14 |
| shared expert gate_up [1280×2560] | 49 | 46.0 | 29.8 | 1.54 |
| shared expert down [2560×640] | 49 | 23.2 | 14.9 | 1.56 |
| router [512×2560] | 49 | 25.6 | 11.9 | 2.15 |
| GDN `in_proj_ba` [96×2560] | 36 | 17.2 | 2.2 | 7.8 |

**Correction to step 2c.** Step 2c's "the single-warp sm80 WMMA kernels are gone, the mixers run at their floor" is
wrong for the prod path. The same `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x{1,2}` kernels serve every
BF16 linear inside the PIECEWISE graphs. 2c's graph timings (33.6–35.5 µs for mixer down) were a different call path;
the in-model kernel takes 40.5 µs.

### Step 3b — standalone BF16 microbench (`tools/bf16mb/`, `notes/data/bf16mb-0925.json`)

M=4, weights rotated over ≥ 96 MiB (never L2-resident), CUDA graph of 48 back-to-back calls. Hypothesis in
`tools/bf16mb/HYPOTHESIS.md`.

| linear | floor | in-model | cuBLAS standalone | best alternative |
|---|---|---|---|---|
| mixer down | 30.2 | 40.5 | 41.5 | Triton split-K (BN16, BK256, S4) **30.6** (1.01×); cuDNN/cuBLASLt via FlashInfer 32.3 |
| mixer up | 29.8 | 33.9 | 39.6 | FlashInfer `mm_bf16` auto **28.7**; Triton 30.4 |
| shared gate_up | 29.8 | 46.0 | 32.8 | FlashInfer tinygemm 29.9 |
| shared down | 14.9 | 23.2 | 15.6 | cuBLAS is best |
| router | 11.9 | 25.6 | 14.9 | Triton 14.3 |
| in_proj_ba | 2.2 | 17.2 | 16.4 | Triton split-K S16 **4.7**; FlashInfer tinygemm 5.3 |

Reading:
- ~~H1 holds for mixer down~~: a second process (`bf16sk_bench.py`, same shapes, same rotation) measured cuBLAS at
  **31.8 µs** for mixer down at M=4, not 41.5, and 29.4 µs for mixer up, not 39.6. Triton gave 30.3 vs 30.6 in the
  two processes, so the bench itself is stable. cuBLAS's kernel choice varies between processes: the first run
  picked `32x32_128x2_align2` plus `splitKreduce`. **Standalone cuBLAS is no reference.** Only the in-model profile
  counts, and there mixer down is 40.5 µs and `in_proj_ba` is 17.2 µs (cuBLAS 16.2–16.4 µs standalone in both
  processes, so that one does reproduce).
- In-model, cuBLAS also launches **~190 `splitKreduce` kernels per step** (9,324 in the trace). They fall in the
  elementwise category, not in the 16.7 ms.
- For the shared expert and the router, standalone cuBLAS sits at 1.05–1.25× floor. Their in-model excess comes from
  running concurrently with the routed MoE on the aux streams, where both share DRAM. That is not a kernel problem.
- H2 holds for mixer down (1.01×). For the router the best alternative only reaches 1.2×.
- **Kernel-swap saving, in-model:** mixer down 100 × (40.5 − 30.6) ≈ 1.0 ms, mixer up 100 × (33.9 − 28.7) ≈ 0.5 ms,
  `in_proj_ba` 36 × (17.2 − 4.7) ≈ 0.45 ms. That is **≈ 2 ms/step ≈ 3.5 %**, the low end of the 2–4 ms expected.
- The first Triton kernel used fp32 `atomic_add` split-K, which is nondeterministic. The deterministic two-pass
  variant (`tools/bf16mb/fn_bf16sk.py`, fixed-order reduction, bitwise repeatable over 20 calls) runs at
  **≤ 1.06× floor on all six shapes at M = 1/4/16** (`notes/data/bf16sk-0925.json`): mixer down 30.3, mixer up 27.2,
  `in_proj_ba` 3.6, router 12.6, shared gate_up 28.5, shared down 14.9 µs at M=4. It is installed env-gated
  (`FN_BF16SK=1`) in the prod venv; the server A/B is `45-bf16sk`.

**Where the 14 ms gap now sits** (at 55.0 ms profiled vs 45.2 floor = 9.8 ms):
- BF16 GEMMs ≈ 5 ms, of which ≈ 2 ms is kernel choice and the rest is MoE-overlap contention and drafter M=1 GEMVs;
- MoE and FP8 at ~1.08× ≈ 2.5 ms;
- idle 2.4 ms;
- GDN update ≈ 0.5 ms.
The remaining lever that no kernel choice can reach is fewer bytes (BF16 leftovers at 8 bits, −8 %).

## Step 3c — why the faster kernel did not help (`47-profsk`, clock check; finding 238)

The server A/B of the deterministic Triton kernel was a per-cycle null (finding 238). A profile of the same prod
config with `FN_BF16SK=1` (`notes/data/profsk-0925-summary.txt`) shows the Triton kernels in-model **run exactly as
slow as the cuBLAS kernels they replaced**:

| linear | cuBLAS in-model | Triton in-model | Triton standalone | floor |
|---|---|---|---|---|
| mixer down [336×10240] | 40.5 | 41.1 | 30.3 | 31.3 |
| mixer up [10240×320] | 33.9 | 33.5 | 27.2 | 29.8 |
| shared gate_up | 46.0 | 42.0 | 28.5 | 29.8 |
| shared down | 23.2 | 17.3 | 14.9 | 14.9 |
| router | 25.6 | 43.1 | 12.6 | 11.9 |
| `in_proj_ba` (+ reduce) | 17.2 | 3.7 + 1.0 | 3.6 | 2.2 |

Only `in_proj_ba` (−0.45 ms/step) and shared down kept their gain; the router got worse. The step went from 55.1 to
55.7 ms profiled.

**Ruled out:**
- **Side-stream contention:** the mixer-down split-K kernel never overlaps an aux stream (0 of 7,039 calls), yet it
  takes 41 µs.
- **Clocks and power** (`notes/data/clk-decode-0925.csv`, 1,500-token c=1 decode on prod, nvidia-smi every 250 ms):
  SM clock 2,528 MHz throughout (2,405 idle), 31 W, temperature 47 °C, throttle reasons `0x0` in all 144 busy samples.
- **The kernel:** two different kernels hit the same in-model time.

**The pattern — per-kernel overhead, not bandwidth.** Big weight reads reach their floor in-model; mid-size ones
carry a roughly fixed extra cost, whatever kernel does the read:

| FP8 GEMM (base profile) | weights | in-model | floor | ratio |
|---|---|---|---|---|
| `in_proj_qkvz` | 42 MB | 187.5 µs | 190.7 | 0.98 |
| QSA `qkv_proj` | 34 MB | 153.9 | 154.9 | 0.99 |
| lm_head | 636 MB | 2,754 | 2,890 | 0.95 |
| `out_proj` / `o_proj` | 15.7 MB | 96.1 | 71.5 | 1.34 |

The BF16 linears (2.6–6.9 MB) carry +6 to +30 µs each. In the step-2c idle in-worker bench, the same weights in the
same process took ~34 µs, so the extra cost appears **only in the live decode sequence**.

Candidates, not yet separated:
- (a) write-back of dirty L2 lines left by the producing kernel (GDN state updates, MoE combine) into the
  consumer's read window;
- (b) DRAM ramp-up at each kernel boundary, which a 48-call standalone chain of the same kernel does not show;
- (c) CPU-side traffic on the unified LPDDR5x during live decode.

**Next instrument:** an in-worker replay of real producer→consumer pairs (e.g. GDN update → `out_proj`, MoE
combine → mixer down) against the consumer alone. The levers that would follow are fewer, larger reads: fusing
router + shared gate_up (same input) and fusing `in_proj_ba` into `in_proj_qkvz`. No new kernel for a mid-size
weight read can win while this overhead holds. Total at stake: the BF16 and `out_proj` excess, ≈ 5 ms per cycle.

**Standalone follow-ups (03:35–03:45)** (`tools/bf16mb/pairbench.py`, `bimodal.py`; data `notes/data/pairbench-0925.json`,
`bimodal-0925.log`):
- **Producer→consumer chain:** it did not separate (a) from (b). The same consumer read 42.1 µs after no producer
  and 30.8 µs after a tiny one, and 26–33 µs after 3/12 MiB dirty writes or a 12 MiB read. The spread was
  measurement state, not the producer.
- **Bimodality, 40 repeats plus 5 after 2 s of sustained load:** unimodal. Triton median 30.9 µs, cuBLAS 31.8 µs,
  every round. Only single replays right after a 0.5 s idle reach 39–43 µs. The "~30 vs ~41 µs LPDDR speed-bin"
  idea is rejected for steady state. A cold first replay after idle does cost ~40 µs, which explains pairbench's
  P0.
- So standalone, both kernels sit at the floor in every warm configuration tried. The in-model ~41 µs appears only
  inside the real decode sequence. **The next instrument is an in-worker replay of the real kernel sequence**
  (FNCAP-style hook, CUDA graph of one decoder layer's actual kernels on its real weights), cut down until the
  extra cost disappears. That is a day job, not a night job.

## Step 4 — looking for the overhead (2026-09-25 morning, "look for overhead" / "maybe we have a scheduling problem")

### 4a. The mid-size-read overhead is tied to one site per layer

Per-call analysis of both whole-step traces (`tools/prof/percall.py`, `byidx.py`, `context.py`, `idlebefore.py`):
- Mixer down is **bimodal**: ~34.6 µs (at floor) at the attention-side site, and 46–53 µs at the MLP-side site,
  which runs right after the attention block's FP8 `out_proj`/`o_proj`. Per-call-index medians correlate at
  r = 0.88 between cuBLAS and Triton in different processes: the site sets the cost, not the kernel.
- The FP8 `out_proj` [2560×6144] itself takes **102 µs after a GDN layer** and **77 µs after a QSA layer**
  (identical shape and kernel; standalone 70.2 µs, floor 71.5).
- **Ruled out:**
  - other-stream work: none overlaps the slow calls;
  - GPU idle before the call: no dose-response within a class, and QSA has more idle yet is faster;
  - GDN state-write volume: the 33-token prefill step, with one state write instead of four, still has the GDN
    `out_proj` at 99.8 µs;
  - the kernel pair itself: standalone `out_proj` → (tiny) → mixer down gives 70.2 / 31.7 µs
    (`tools/bf16mb/pairbench2.py`, `notes/data/pairbench2-0925.json`).
- Cost: ≈ 1.1 ms (GDN `out_proj`) + ≈ 0.6 ms (mixer down at that site) per step. Mechanism still open.

### 4b. Small kernels
1,672 kernels per step under 5 µs (3.0 ms/step) and 454 of 5–20 µs (4.3 ms/step). The biggest groups:
- aten elementwise 0.74 ms;
- the hyper-connection helpers (`_hc_combine_norm`, `_hc_gate_mix`, `_hc_silu`) 0.5 ms;
- cuBLAS `splitKreduce` 0.22 ms;
- FP8 act-quant 0.18 ms;
- MoE routing/prefix-sum kernels ~0.5 ms.
These are fusion candidates.

### 4c. The "restart drift" is 11 %, is over within ~1 minute of decoding — every A/B measures it
The DS4.1 lesson was to check whether the device waits on the host (`deepseek-moe-gb10/notes/early-submit-profile.md`,
`ds41-decode-bottleneck-2026-09-16.md`). On warm prod (up 4.5 h), the same probe gives **53.6–53.9 ms/step** for
both decode(150) and decode(400). The first request after hours of idle ran at 56.6. `tools/prof/lenprobe.py`.

| | ms/step |
|---|---|
| fresh server, first 400 tokens after 2×64 warm-up (40-prof, 47-profsk probes) | 59.6 / 59.8 |
| same server a minute later, under the profiler | 55.0 / 55.6 (profiler ≈ +1.3 ms) |
| warm prod, unprofiled | **53.7** |

- So the ~6 ms/step "warm-up drift" ends within the first few hundred decode tokens. It is not a 15-minute
  effect.
- Every armrun A/B measures its c=1 cell in that transient: decprobe's c=1 runs first, at 59 ms/cycle.
- Warm prod does not page: 9 worker/engine major faults in a 10 s decode (`tools/prof/faults.py`). All 47.7 GiB of
  `model-plefp8-*` shards are resident (`tools/prof/resident.py`).
- Open question: in the cold window, is the GPU waiting on the host (first-touch PLE faults from NVMe, allocator,
  Python warm-up) or are the kernels slower? Next instrument: profile the first 400 tokens of a fresh server, and
  log worker majflt per step, against the warm profile.

### 4d. DS4.1 lessons checked against Flash-Next (`deepseek-moe-gb10`, `dsv41-enginev2` notes)
| DS4.1 finding | Flash-Next, measured in the prod trace | verdict |
|---|---|---|
| First decode of a process slower, "tracking the Engram read time and nothing else" (cold Engram row cache) | the 11 % cold window of 4c; our PLE is the same Engram mechanism, read from mapped checkpoint pages | **same mechanism likely**; test = fresh-start fault log + a pre-warm arm |
| Up to 11 host syncs per step → one 7-element pinned readback ("lean step") | 5 D2H + 16 H2D pinned + 47 small D2D copies per step, 0.1 ms GPU; no sync storm | clean, no lever |
| Weights read twice per step (LM head in verify + draft; engram `wkv` twice) | target weights and FP8 lm_head once per verify. Drafter BF16 dense once per draft pass: qkv ~81 MB (M=1 `gemvx`, 2 per step), hc-collapse [2560×10240] 52 MB (`gemv`, 3 per step) | not duplicated, but **~1.8 ms/step of BF16 drafter reads**; DS4.1's fix was a narrower stored format (FP8 head, quality-neutral) → FP8 drafter dense ≈ −0.9 ms/step |
| "The 32 % slowdown was a host wall clock" (measurement artifact) | PDL double-counting checked: same-stream overlap 0.01 ms/step, kernel durations are real | ruled out |
| "A mechanism is a hypothesis until something measures it" | the cold window = PLE first-touch is still unmeasured | open |

### 4e. The cold window is PLE paging, and a fresh server cannot escape it (2026-09-25, prod down; `tools/plecold/`)
Probe: `tools/plecold/coldprobe.py`, 6 different 300-token requests per fresh start, PLE page-cache residency (mincore),
worker major/minor faults, NVMe reads. Data `notes/data/{plecold,plekv4,pledrop}-0925.jsonl`, hypotheses in
`tools/plecold/HYPOTHESIS.md`.

| config (fresh start) | PLE resident at ready → after warm | major faults/step | ms/step (req 0 … 5) |
|---|---|---|---|
| prod config (KV 33.5 GiB) | 0.03 GiB → — | 27–32 | 60.1 … 58.7 |
| + FN_PLE_POPULATE at init | 0.02 (populated at 08:28, evicted by KV/graph init) | 26–36 | 60.1 … 59.0 |
| KV 4 GiB | 0.03 → — | 29–36 | 60.4 … 58.7 |
| KV 4 GiB + warm after ready | 0.03 → 16.7 | 15–26 | 59.0 … 57.5 |
| KV 2 GiB + FNDROPCACHE + warm | 13.8 → 20.7 | 16–23 | 59.0 … 57.6 |
| warm prod (hours up) | 47.7 (100 %) | ~0 | 53.7 |

- **Loading evicts the table.** It streams ~72 GB of weight shards through the page cache.
- **KV size makes no difference at ready.** Page cache then caps at ~32 GiB; MemAvailable ~33 GiB with
  72.4 GiB weights, 7.4 GiB anon and ~5 GiB driver/slab. So at most ~43 % of the 47.7 GiB table fits.
- **Faults barely fall with traffic.** Rows are hash-spread, so new tokens touch new pages.
- **Cost:** ≈ 0.2 ms per major fault per step, linear across the configurations. That is 6 ms/step (11 %) in the
  prod-config restart.
- **Consequence for earlier measurements.** Every armrun A/B (findings 233–238) ran in the ~30-faults regime. The
  stall is roughly constant per step in both arms, so relative results are expected to hold and absolute
  ms/step are ~11 % high. The FNBF16SK null is not explained by an additive stall.
- **Open.** Warm prod reached 100 % residency while `free` showed only 3 GB buff/cache, so its PLE pages were not
  ordinary page cache. How prod gets there, and whether a restart can reach it quickly, is the lever: 53.7 vs
  ~60 ms/step.

### 4f. Who takes the PLE faults: the GPU, serially, on every step (FN_PFTIME, `tools/plecold/ple_pageable_pftime.py`)
**Fault latency on the idle box** (`tools/plecold/faultlat.py`, `faultpar.py`):
- One random 4 KiB read ≈ 60 µs by any route: O_DIRECT 64, buffered 58, mmap fault 58 µs. A cached minor fault is
  1.6 µs. The OS path adds nothing.
- In parallel the SSD gives ~50k IOPS; the prefetcher's numpy touch pattern scales to 231k rows/s at 64 threads.

**In the server** (1 cold start, KV 4 GiB, 700 steps, `notes/data/plepf-0925-summaries.txt`), per step, p50:

| | |
|---|---|
| prefetch `prepare()` → inputs visible on the host | **~62 ms**, one whole step: the ids come from the previous step's GPU sampling |
| host touch after that | 3.7–5.4 ms |
| GPU gather kernel (`_gather_mapped_rows_kernel`) | **3.7–5.5 ms** (p90 5.3–9.3, max 52 ms) |
| GPU started the gather before the touch finished | **100 % of steps** |
| major faults during the touch window | 27–32 |

- **The prefetch can never win.** The rows of step N depend on step N's token ids. Those exist only when step N−1's
  sampling finishes on the GPU, and step N's forward starts the PLE gather immediately. There is zero slack.
- **So the GPU faults the ~30 pages itself, one at a time**, at ~150 µs each (60 µs SSD + ~90 µs GPU fault
  handling): 4–5 ms/step. The CPU touch runs concurrently and waits on the same page reads. In the cold state the
  prefetcher is dead weight.
- **Fix directions**, none built:
  - (1) keep the table resident: how does warm prod reach 100 % when a fresh server caps at ~43 %?
  - (2) replace the GPU's serial ATS faults for missing rows with parallel host reads (~30 rows at 50k IOPS
    ≈ 0.6 ms) behind a GPU wait;
  - (3) make the table fit: an FP4 PLE (~24 GiB) would stay fully resident beside the weights, but costs quality.

### 4g. Option 2 (the lookup waits for a parallel CPU fault-in): null (FN_PLE_SYNCTOUCH, 2 rounds, 1 cold start/arm)
The user's direction: the PLE must not take KV memory, so fix the cold faults in the lookup path instead of making
the table resident. `tools/plecold/ple_pageable_synctouch.py`: `gather_into` waits (≤ 50 ms, GIL released) until
the prefetch thread has faulted the step's pages.
- **Round 6** (stock touch): outputs identical 6/6, GPU gather 4 ms → 0.11 ms. But ms/step got **worse**
  (61–64 vs 59–61). The stock `touch()` handles < 4096 rows serially in one thread: 5.6 ms per decode step. The
  true fault count is ~56/step; base had counted only the CPU's ~25.
- **Microbench, 56 cold rows** (`smalltouch.py`): serial 4.8 ms; numpy split into 16 tasks 1.66 ms; per-page
  `MADV_POPULATE_READ` on 32 tasks 1.37 ms ≈ 40k IOPS, near the SSD's parallel ceiling.
- **Round 7** (per-page POPULATE_READ, 32 tasks): outputs identical 6/6, gather 0.10 ms. But the touch takes
  **3.7 ms in the server**, ~15k IOPS, so ms/step is 58.6–59.5 vs base 58.0–60.9: **null**. The wait moves the stall
  from GPU to CPU without shrinking it.
- **Suspect** (unmeasured): direct reclaim in the fault path. In the server MemFree is ~3 GiB, against 117 GiB in
  the microbench, so every new page first evicts another.
- **Data:** `notes/data/plesync-r{6,7}-0925*.jsonl`, `plesync-r7-0925-summaries.txt`.

### 4h. Round 8: no reclaim. Round 9: the GIL was the bottleneck, and the C helper wins −2.2 ms/step cold
- **Round 8** (`notes/data/plerecl-r8-0925.jsonl`): pgscan_direct, pgsteal_direct, allocstall and pgscan_kswapd are
  all 0 per step; MemFree 6.8 GiB. Memory pressure is ruled out.
- **Microbench:** the Python pool doing per-page `madvise` takes 1.8 ms for 57 pages. `tools/plecold/fnpopulate.c`,
  one ctypes call that populates from 16 pthreads without the GIL, takes **0.60 ms** (~95k pages/s).
- **Round 9** (sync touch through the C helper, `tools/plecold/ple_pageable_synctouch_c.py`; 1 cold start per arm,
  KV 4 GiB; `notes/data/plesync-r9-0925*.jsonl`/`summaries.txt`):

| request | base ms/step | sync+C ms/step | Δ |
|---|---|---|---|
| 0–5 | 60.38 / 60.03 / 60.29 / 59.91 / 59.90 / 57.93 | 57.97 / 57.45 / 58.07 / 57.16 / 57.57 / 56.90 | −2.41 / −2.58 / −2.22 / −2.75 / −2.33 / −1.03 |
| mean | 59.74 | 57.52 | **−2.2 (−3.7 %)** |

  - Outputs identical 6/6.
  - In-server touch p50 2.0 ms (3.7 ms with the Python pool); GPU gather 80 µs (base 3.7–4.8 ms).
- **Not yet established:**
  - only one start per arm;
  - the warm state is unmeasured (resident pages; the wait still costs the host its one-step lead);
  - the in-server touch is 3× the idle-box microbench.

### 4i. Round 10 (2 starts per arm, cold pass + warm second pass): the wait wins cold, loses warm
`notes/data/plesync-r10-0925.jsonl`; KV 4 GiB. The warm pass repeats the same 6 prompts, so ~0 major faults per step.

| ms/step | base, start 0 / 1 | sync + C, start 0 / 1 | Δ |
|---|---|---|---|
| cold pass (6 requests) | 59.26 / 59.87 | 57.52 / 57.52 | **−2.05**; faster on 12/12 paired requests |
| warm pass | 54.78 / 54.98 | 57.20 / 57.20 | **+2.3** |

- Output hashes are identical in all 24 requests.
- **Warm fresh-server baseline: 54.9 ms/step**, close to warm prod's 53.7. The speed-of-light gap is therefore ~10 ms,
  not the 14 ms measured in the paging regime.
- The unconditional wait costs ~2.3 ms/step whenever nothing faults: the host gives up its one-step lead at every
  lookup. Next: the auto mode, which waits only while ≥ 12 faults/step are measured (`FN_PLE_SYNCTOUCH=auto`).

### 4j. Round 11: the auto-gated wait keeps the cold gain and drops the warm penalty
`FN_PLE_SYNCTOUCH=auto` (`tools/plecold/ple_pageable_synctouch_auto.py`): the lookup waits only while the EMA of
measured major faults per step is ≥ 12. The EMA starts high (waiting) and the C populate helper is used.
1 start per arm (`notes/data/pleauto-r11-0925*`):

| ms/step | base | auto | Δ |
|---|---|---|---|
| cold pass | 60.02 | **57.46** | **−2.56**; faster on 6/6 requests |
| warm pass | 54.88 | 55.20 | +0.32 (unconditional wait: +2.3) |

- The gate switched as designed: fault EMA 51–61 per step in the cold pass (waiting); **0.0** in the warm pass, and
  no waits there. Outputs identical 6/6.
- **Next:** confirm with a second start pair, then the follow-up PR on top of #58439, with the populate helper as a
  `csrc/` CPU op and the fault-rate gate.

**Confirmed, second start pair** (`notes/data/pleauto-r12-0925.jsonl`):
- cold: base 59.84, auto 57.72 → **−2.12** ms/step;
- warm: base 54.98, auto 55.17 → +0.19.

Across both pairs: cold −2.56 / −2.12, warm +0.32 / +0.19; outputs identical in every request. The auto-gated
wait is the candidate for the follow-up PR on top of #58439.

### 4k. Warm profile at 4 GiB KV (`profwarm`): the overhead map stands
Same probe as 40-prof: the prompt is warmed first, so its PLE rows are cached. `notes/data/profwarm-0925-summary.txt`.

| | 40-prof (default KV) | profwarm (KV 4 GiB) |
|---|---|---|
| profiled step / GPU busy | 55.1 / 52.7 ms | 55.4 / 53.0 ms |
| GDN `out_proj` / QSA `o_proj` | 102 / 77 µs | 101.2 / 75.6 µs |
| mixer down after out/o_proj vs elsewhere | 46.7–49.5 vs 34.6 µs | 47.7–50.4 vs 34.5 µs |
| BF16 wmma, FP8 blockwise, MoE grouped (ms/step) | 14.6, 16.3, 18.2 | 14.8, 16.1, 18.4 |
| PLE gather | — | ~13 µs per call |

- 40-prof was already effectively warm (its probe repeats one prompt), so steps 3, 3c, 4a and 4b are **not** paging
  artifacts.
- The post-attention slow sites (~1.7 ms/step), the small-kernel budget (~7 ms), the MoE-overlap contention and
  the 2.4 ms idle are in-model effects of the warm steady state.
- Unprofiled warm: 54.9 ms/step, i.e. **~9.7 ms above the 45.2 ms floor**.

### 4l. FNBF16SK re-tested warm: the null stands (finding 238 addendum)
c=1 per cycle 55.62 vs 55.76 ms (+0.25 %); c=4 +0.9 % per cycle. Paging did not mask kernel-level gains, so the
decisive test says no broad re-runs are needed. Absolute numbers from the paging regime are 6–8 % slow.

### 4m. The small-kernel bucket on the critical path is 3.5 ms/step, not 7.3 (warm trace, `tools/prof/gaps.py`)
- Most small kernels in the MoE blocks (shared expert, `act_and_mul`, `_hc_combine_norm`) run on the main stream
  while the routed experts run on the aux stream (~400 µs/layer). They are hidden, and fusing them saves nothing.
- **Main-stream time not overlapped by a side stream:** < 5 µs kernels 2.11 ms/step, 5–20 µs 1.42 ms/step, ≥ 20 µs
  28.5 ms/step.
- **Critical-path small kernels:**
  - small BF16 GEMMs 0.86 ms (the FNBF16SK Triton path already cuts `in_proj_ba` by −0.45);
  - hc helpers (`combine_norm`, `gate_mix`, `silu`, ~106 each) 0.52;
  - aten elementwise glue (~490) 0.52;
  - FP8 activation quant (97) 0.18;
  - cuBLAS `splitKreduce` (119) 0.16;
  - GDN conv update + norm + D2D copy 0.29;
  - QSA sparse kernels 0.37 (real work).
- **GPU idle (union of streams): 1.8 ms/step.** About 0.6 ms is launch gaps in the eager GDN section (~6 gaps per
  layer × 36 layers), about 0.2 ms in the QSA section.
- **Realistic fusion prize: ~1–1.5 ms/step (2–3 %)**, spread over 3–4 separate kernel fusions:
  - `_hc_combine_norm` + `per_token_group_quant`;
  - `hc_gate_mix` + `silu`;
  - the elementwise glue in the eager GDN section;
  - no `splitKreduce`.
- The largest single open item remains the post-attention slow sites, ~1.7 ms/step, mechanism unknown.

### 4n. Slow-spot hunt: graph position, TLB, clocks ruled out; first-in-graph kernels are the lead
- **Graph position** (warm trace, `tools/prof/graphpos.py`): not the cause alone.
  - Mixer down elsewhere is fast even at graph positions 1–2 (34.1–34.7 µs).
  - After `out_proj` it is slow at positions 3 and 5 (50.4 / 47.7 µs); `out_proj` itself sits at positions 1–3 and
    takes 99.4 µs.
  - But a BF16 GEMM that normally takes 23 µs takes **177 µs as the first kernel of a graph launch** (once per step),
    and mixer down at positions 1–3 averages 49.8 vs 35.6 µs later.
- **TLB** (`tools/bf16mb/tlbbench.py`): sweeps of 6,144 distinct 2 MiB regions over 12 GiB before the consumers had
  no effect.
- **Clocks / power** (`dvfsbench.py`, default vs `nvidia-smi -lgc 2418`, reverted with `-rgc`):
  - identical consumer times;
  - SW power-cap time 1,354 ms in the default run and 0 ms locked.
  The GB10 does power-cap under load (~2.8 h of SW power capping accumulated since boot), but that is not the slow
  spots.
- **Correction:** the standalone "busy vs idle producer" effect (out_proj 74 vs 84 µs) is host launch latency inside
  eager event timing. It is not relevant to the in-model CUPTI kernel durations.
- **Next:** an nsys trace with `--cuda-graph-trace=node` of one warm server, to see graph-launch-side work
  (upload, memory-pool mapping) that is billed to the first kernels of a launch.

### 4o. SOLVED: the post-attention slow spots are the write-back of the GDN spec-decode state snapshots
- **nsys node trace** (warm, `--cuda-graph-trace=node`, `tools/prof/nsysan.py`):
  - GDN `out_proj` 102.0 µs and QSA `o_proj` 75.7 µs, back-to-back (0.5 µs gap);
  - graphs are launched ~75 ms before their kernels run, so graph-launch work is ruled out.
- **Directly before GDN `out_proj`:** `fused_sigmoid_gating_delta_rule_update` (42 µs, grid [1,4,48]). In spec decode it
  stores the full fp32 state after **every** one of the T=4 verify tokens into its own cache slot
  (`fused_sigmoid_gating.py` lines 155–170): 4 × 48 heads × 128 × 128 × 4 B = **12 MiB per GDN layer per step**,
  written in 42 µs, faster than DRAM. The lines sit dirty in the 24 MiB L2, and the next kernels pay their
  write-back.
- **Reproduced standalone** (`tools/bf16mb/l2dirty.py`, CUDA graph, CUPTI durations; `notes/data/l2dirty-0925.json`):

| dirty L2 before the pair | out_proj | mixer down |
|---|---|---|
| 0 MiB | 71.4 | 31.2 |
| 3 MiB | 74.9 | 37.8 |
| 6 MiB | 81.6 | 42.4 |
| **12 MiB** | **99.2** | **45.5** |
| in-model after a GDN layer | 99–102 | 47.7 |

- **Cost:** ~40 µs per GDN layer × 36 ≈ **1.5 ms/step**.
- **Waste:** the next step reads only the state after the last accepted token (`num_accepted_tokens − 1`). On average
  ~2.5 of the 4 snapshots (~7.5 MiB per layer) are written and never read. Writing one snapshot would save ~30 µs per
  layer ≈ **1.1 ms/step (~2 %)**.
- **Upstream:** vllm#49887 **ReplaySSM** (Dao AI Lab + NVIDIA, open, stacked on #49847) replaces the T snapshots with one
  checkpoint plus a ring of per-token inputs (fp16 d/k, fp32 g). It integrates the Qwen3.5 GDN mixer only, and its
  thread raises prefix caching as a requirement. Our dirty-L2 aftershock is extra evidence for it on unified-memory
  GB10. Not tested on Qwen4Exp.
- **Earlier wrong turns**, for the record: graph-launch position, TLB, clocks/power (4n); my first dirty-L2 test
  (`pairbench.py` P3), whose 110 µs producer drained its own writes; and the prefill counter-check (4a), where the
  chunked prefill kernel writes differently.

### 4p. Streaming stores for the snapshots (FNGDNCS, `.cs`): null
`tools/gdncs/`; warm decprobe (second pass), KV 4 GiB, 2 starts per arm (`notes/data/gdncs-0925.jsonl`):
- c=1 per cycle: base 55.70 / 55.69, cs 55.74 / 55.60 ms → **Δ −0.02 ms**;
- c=4 per cycle: 105.14 (base start 1) vs 104.95 / 104.71 → −0.3 %. Base start 0's c=4 produced different text,
  hash `2671b9…`, and is excluded.
- Output hashes identical, as expected: values are unchanged.
- The cache hint does not avoid the aftershock, so only fewer snapshot bytes can: ReplaySSM (#49887), or narrower
  snapshots. Next: `--mamba-ssm-cache-dtype float16` (halves them to 6 MiB/layer), speed first, then quality.

### 4q. fp16 SSM state cache (`--mamba-ssm-cache-dtype float16`): −2.2 % c=1, −4.4 % c=4 per cycle — quality pending
Halves the GDN snapshots (12 → 6 MiB per layer per step). The kernel still computes in fp32; only the stored states
and the loaded initial state are fp16. The model default is `mamba_ssm_dtype: float32`. Warm decprobe, KV 4 GiB,
2 starts per arm (`notes/data/ssm16-0925.jsonl`):

| per cycle | base (start 0 / 1) | fp16 (start 0 / 1) | Δ |
|---|---|---|---|
| c=1 | 55.70 / 55.59 ms | 54.39 / 54.46 ms | **−1.22 ms (−2.2 %)** |
| c=4 | 105.38 / 105.09 ms | 100.83 / 100.54 ms | **−4.4 %** (4 sequences, 4× the snapshot bytes) |

- fp16 output is deterministic across starts (same hash) but differs from fp32. c=1 acceptance moved 2.535 → 2.424
  with the changed text, so c=1 tok/s alone is not the metric.
- **Not a promotion candidate until quality is measured.** The decode-time divergence probe
  (`tools/gdncs/divprobe.py` / `divcmp.py`) compares base vs fp16 vs a known-benign reference: FNBF16SK, which only
  changes reduction order.

### 4r. fp16 SSM state cache: quality — no drift, but a real per-token perturbation (user decision, not promoted)
- **Short horizon** (`divprobe.py`, 8 prompts × 512 greedy tokens):
  - fp16 first diverges at a median of 24.5 tokens; mean |Δlogprob| before divergence 0.037 (p99 0.56);
  - the reduction-order reference (FNBF16SK) diverges at 31 tokens, 0.023 (p99 0.42).
- **Long horizon** (`lpprobe.py`/`lpcmp.py`): 3 documents of 5–11k tokens, teacher-forced prompt logprobs,
  `FN_BATCH=256`, so the state is stored and reloaded every 256 tokens; no prefix cache. Mean |Δlogprob| vs fp32:

| doc | tokens | by position quarter | NLL change |
|---|---|---|---|
| a | 11,219 | 0.227 / 0.187 / 0.260 / 0.255 | −0.06 % |
| b | 4,978 | 0.231 / 0.313 / 0.300 / 0.286 | −0.57 % |
| c | 9,688 | 0.135 / 0.213 / 0.174 / 0.190 | +0.07 % |

- **Flat across position** (no compounding) and **NLL-neutral**. Per-token size ~0.13–0.31 nats is ~10–20× the
  FP8-vs-BF16 reference (0.014) and ~1/5 of the full NVFP4-vs-FP8 gap (1.11, `nvfp4-quantization-cost-measured`).
  b.md's lower NLL could be softening (memory `lower-nll-can-be-softening`).
- The reference arm is exactly 0 here: FNBF16SK engages only at M ≤ 16, so prefill chunks take the stock path. There
  is no noise floor from this run.
- **Verdict: a quality trade, not free speed.** −2.2 % c=1 / −4.4 % c=4 against a per-token perturbation the size of a
  fifth of the 4-bit step. Not promoted; the user decides. A task-level eval would be needed (SWE-bench resolves
  only ~12 pp, so it cannot see this).
- **The principled path remains ReplaySSM (#49887):** fp32 checkpoint plus an fp16 input ring, so no stored-state
  precision loss. It needs a port to Qwen4Exp.

### 4s. RecoverSSM for the GDN layers, phase 1 (no prefix caching): −2.2 % c=1, −5.0 % c=4 per cycle, fp32 state kept
- **What:** the Kimi-K3 KDA RecoverSSM protocol ported to Qwen4Exp's GDN layers (`vllm-venv-rssm`, a clone; env gate
  `FN_GDN_RECOVERSSM=1`). Verify runs from the checkpoint state and stores only a per-token (correction, k, g) record,
  (48, 4, 257) fp32 per layer. After sampling, one commit kernel replays the accepted tokens and writes the state
  once. This replaces the four full fp32 state snapshots per step that §4o blamed for the slow spots.
- **Kernel test** (`test_recoverssm_gdn.py`): verify outputs bit-identical to the native kernel; committed states
  within 1.4e-7 relative, for 1–4 accepted tokens.
- **Server A/B** `rssm1c`: same venv both arms, MTP n=3, KV 4 GiB, `--no-enable-prefix-caching`, 2 starts per arm,
  alternating. Path lines required: `FNRSSM: GDN RecoverSSM speculative verify active`, `GDN RecoverSSM path taken`
  (rssm); base must show neither and bind 2 states. Data: `data/rssm/`.

| arm | c=1 ms/tok | accept len | c=1 ms/cycle | c=4 tok/s | c=4 ms/cycle |
|---|---|---|---|---|---|
| base (2 starts) | 21.869–21.89 | 2.535 | 55.44–55.49 | 93.80–93.86 | 104.84–104.90 |
| rssm (2 starts) | 21.44–21.46 | 2.53 | 54.24–54.29 | 99.99 | 99.61 |
| Δ | −2.0 % | | **−1.2 ms (−2.2 %)** | +6.6 % | **−5.3 ms (−5.0 %)** |

- Inside the hypothesis written before the run (`data/rssm/HYPOTHESIS.md`: c=1 −0.8…−1.5 ms, c=4 −2…−5 %). c=4 gains
  more because four rows' snapshots were four times the dirty L2 write-back.
- **Correctness** (`divprobe.py`, 8 prompts × 512 greedy tokens):
  - rssm vs base: first divergence median **26 tokens**; mean |Δlogprob| before it 0.026 (p99 0.29).
  - The reduction-order reference (FNBF16SK) diverges at 31 / 0.023, and the fp16 SSM cache (§4r) at 24.5 / 0.037.
  - So RecoverSSM perturbs like a summation-order change, less than fp16 storage. The state stays fp32; the
    difference is the replay's accumulation order.
  - Each arm reproduces its hashes across restarts (base1 = base0, rssm1 = rssm0). Acceptance is unchanged
    (2.53 vs 2.535). All 8 texts read correctly.
- The agent-loop probe's s/turn (3.25 → 2.99) is not comparable: the two arms generated different token counts
  (189 vs 124).
- **Limits:** phase 1 needs `mamba_cache_mode none`, and prod runs align with prefix caching. PIECEWISE graphs only
  (the builder refuses FULL decode graphs). Phase 2 (align, with the PLE short conv on the same protocol) is next.

### 4t. RecoverSSM phase 2 (align + prefix caching, the prod mode): −2.4…−4.0 % c=1, −5.2 % c=4, agent loop −16 %/turn
- **What:** phase 1 plus `patch_rssm2.py`. Align mode is allowed on the V2 runner, and the PLE short conv follows the
  same protocol (`ple_recoverssm.py`: one block, compacted conv window, no per-draft blocks). Prefix caching is on,
  so vLLM picks `mamba_cache_mode align` itself, as in prod.
- **A/B** `rssm2b`: same venv, MTP n=3, KV 4 GiB, 2 starts per arm, alternating. The probe (`comboprobe2.py`) runs
  the decode set twice, and the second pass hits the prefix cache.
- **Path lines required on the rssm arm:** `GDN RecoverSSM path taken … align=True` and
  `FNRSSM PLE RecoverSSM path taken`. Data: `data/rssm/rssm2b*`.

| arm | c=1 ms/tok | c=1 ms/cycle | c=4 tok/s | c=4 ms/cycle | agent s/turn | agent ms/tok | prefix hit | KV tokens (4 GiB) |
|---|---|---|---|---|---|---|---|---|
| base-align | 21.976–22.347 | 55.71–56.65 | 93.35–93.65 | 105.07–105.41 | 1.31 | 44.53–44.57 | 77.4 % | 75,678 |
| rssm-align | 21.491–21.493 | 54.37–54.38 | 99.69–100.01 | 99.59–99.91 | 1.10 | 39.38–39.39 | 82.1 % | 103,953 |
| Δ | −2.2…−3.8 % | **−1.3…−2.3 ms** | +6.5…+7.1 % | **−5.2 %** | **−16 %** | −11.6 % | +4.7 pp | **+37 %** |

- **Decode:** as in phase 1, with a slightly larger c=1 gain. Base-align's start-to-start spread (1.7 %) is the
  bigger uncertainty. Inside the phase-2 hypothesis.
- **Cache-hit replay is self-consistent:** in every arm and start, the cache-hit pass reproduced the first pass's
  hashes (8/8). RecoverSSM-align's decode-probe hashes equal phase 1's, and so do base-align's and phase-1 base's, so
  align mode changes nothing. Divergence vs base-align is unchanged: median 26 tokens. In the divergence probe,
  cross-phase pairs agree to 512 tokens except the prompts that also vary between two restarts of the same arm
  (#3 at ~95–100 tokens, #4 at 408 in rssm). That is existing restart nondeterminism, not a mode effect.
- **The agent loop gains three times the decode gain.** Two causes can be read off the logs, but neither is
  decomposed:
  - no per-draft Mamba blocks (the old spec-decode block cost, memory `spec-decode-prefix-cost-agentloop`), so
    the prefix hit rate rises 77.4 → 82.1 %;
  - the same 4 GiB holds 37 % more KV tokens, because a request's Mamba page no longer carries speculative
    blocks.
  - The token counts are close (224 vs 236), so ms/tok (−11.6 %) is the fair number.
- **What it takes to reach prod** (needs the user's go): the three phase-1 patches, `patch_rssm2.py` and the three
  new modules, installed into the prod venv behind `FN_GDN_RECOVERSSM=1`. PIECEWISE graphs only, which prod
  already uses.
- Not measured: long contexts (> 8k) under RecoverSSM; c ≥ 8; a long-horizon quality check. Prompt-logprob
  probes run prefill only and never take the verify path, so they cannot see this change.

### 4u. Node trace, base-align vs RecoverSSM-align: the slow spots are gone; the commit is at the DRAM floor
Warm nsys node traces, one ~150-token c=1 request per arm. Setup as in §4t, with the §4o recipe. Compared by
`tools/prof/nsyscmp.py` (the window trimmed 10 % at each end; steps = GDN spec kernels / 36). Reports in
`/opt/llm/capture/nsys-0926/`, output `data/rssm/nsys0926-cmp.txt`.

| per step (ms) | base-align | rssm-align | Δ |
|---|---|---|---|
| step | 55.53 | 54.23 | **−1.29** (A/B §4t: −1.3) |
| GPU busy (all streams) | 53.03 | 51.68 | −1.35 |
| idle | 2.49 | 2.55 | +0.06 |
| FP8 blockwise GEMM | 16.07 | 14.93 | −1.14 |
| BF16 GEMM/GEMV (mixer, lm_head) | 16.76 | 16.24 | −0.52 |
| GDN spec kernel (fused update → verify) | 1.55 | 0.85 | −0.71 |
| RecoverSSM commit (3 launches, 12 layers each) | — | 1.05 | +1.05 |
| NVFP4 MoE | 20.69 | 20.61 | −0.09 |

- **H1 held:** GDN `out_proj` median 101.4 → **73.8 µs** (the §4o standalone clean value is 71.4). QSA `o_proj` is
  unchanged (76.7 / 75.9). The write-back aftershock is gone, and it accounts for the GEMM deltas.
- **H2 missed twice:**
  - The verify kernel got 45 % faster, not ±15 %: it no longer writes state.
  - The commit costs 1.05 ms, not ≤ 0.3. It must read and write the whole fp32 state of every layer:
    72 MiB per 12-layer launch in 330 µs = **224 GB/s, the DRAM floor.** My estimate had counted only the replay
    records.
- **H3 held:** idle is unchanged. The rest of the overhead map (§4k/§4m) does not move.
- **What is left on the GDN state path:** verify (read 3 MiB per layer) plus commit (read + write) moves 324 MiB
  per step, 1.54 ms at 220 GB/s. The floor, one read and one write, is 216 MiB = 1.03 ms. Two levers:
  - **(a) Deferred commit:** fold step N's commit into step N+1's verify (load the checkpoint, replay the
    accepted records, store the new checkpoint, then verify the drafts). This is ReplaySSM's layout. It removes
    one full state read, about **−0.5 ms/step (~1 %)**. It needs the replay record to survive a step and care
    at request finish and at align-mode block boundaries.
  - **(b) Verify bandwidth:** the verify reads at 145 GB/s (23.1 µs per layer). At the floor it saves
    **~0.25 ms/step**; try BV/num_warps first.
- The big buckets are unchanged and all GEMM: NVFP4 MoE 20.6, BF16 16.2, FP8 14.9 ms/step.

### 4v. Verify-kernel launch config: null (standalone, `tools/rssm/verifybench.py`)
BV ∈ {4…64} × num_warps ∈ {1, 2, 4, 8}. The L2 is flushed before each call, and timings are CUDA-graph replays at
T=4. Every config's output and replay record were compared with the shipped config (BV 32, 4 warps). Data:
`data/rssm/verifybench-0926.json`.
- **Batch 1:** the shipped config, 21.2 µs (148 GB/s; in-model 23.1), is already the fastest bit-identical one.
  - Faster configs save at most 2.4 µs per layer (BV 8, 1 warp: 18.8 µs), 0.09 ms/step. All of them change the
    output: the warp count changes the reduction tree over K.
- **Batch 4:** BV 8 with 4 warps is bit-identical and 4 % faster (68.5 vs 71.4 µs).
- **Below the pre-set bar** (≥ 4 µs/layer at batch 1), so there is no server A/B. The kernel is latency-bound: each
  CTA runs a 4-token dependent chain of two K-reductions on a 16 KiB tile, and bandwidth is not the limit. The
  hypothesis (15–18 µs) missed.

### 4w. Deferred commit (fold the commit into the next verify): designed, not built — ~0.9 % for a cache-correctness risk
**Gain:** it removes one full fp32 state read per step. That is 108 MiB ≈ **0.5 ms/step at c=1, ~0.9 %** (§4u). The
state path then sits at its floor, one read plus one write.

**Mode `none`, tractable:**
- The replay record lives in the request's own page slot, and the V2 runner already passes step N's accepted count
  into step N+1's metadata (`num_accepted_tokens`).
- The verify kernel would load the checkpoint, replay `a_N` records, store the checkpoint, then verify the new
  drafts.
- Needs (1) a per-slot "records valid" flag that the prefill path clears, so a row fresh from prefill replays
  nothing.
- Needs (2) per-v-slice copies of k/g in the record, because today only `pid_v == 0` writes them, so a sibling CTA
  would read the next step's k.

**Mode `align` (prod), hazardous:**
- (3) The commit plan (final and boundary block, recovery lengths) must persist per request across batch reorder
  (`idx_mapping`) and across the block moves at `mamba_block_size` boundaries.
- (4) A finished or preempted request's pending commit must be flushed before its blocks can serve a prefix-cache
  hit in the very next step's prefill.
- (5) Any other reader of the state between steps (chunked prefill of a continuing request, block copies) must see
  a committed state.
- A mistake in (3)–(5) silently corrupts cached states. `comboprobe2`'s cache-replay check covers only a part of
  that.

**Decision:** not built tonight; it is ~0.9 % against that risk and needs the user's priority call. If built: mode
`none` first, to measure the real gain; align only if it holds; the kernel test extended to replay-across-steps
and flush; the server A/B with `comboprobe2` plus an agent-loop hash check.

**PROD 2026-09-26 (the user's go, "yes 1..3"):** RecoverSSM is installed into the prod venv `vllm-venv-main1ea7` with
the four patch scripts; backups are `*.orig-rssmprod`. It is enabled by drop-in `35-recoverssm.conf`
(`FN_GDN_RECOVERSSM=1`).
- **Validation `rssmprod`** on the prod venv, 1 start per arm, same probe as §4t:
  - flag off reproduces the stock hashes (`c0c061e6…`, 21.93 ms/tok);
  - flag on reproduces the clone's hashes (`2f3574ed…`, 21.57 ms/tok, c=4 99.6 tok/s, agent 1.11 s/turn);
  - the installed files are byte-identical to the tested clone's, minus the clone-only debug line.
- The service stays stopped until the user asks for it up (it was down by the user's order).
- **Revert:** `rm /etc/systemd/system/vllm-flashnext.service.d/35-recoverssm.conf && systemctl daemon-reload`. With
  the flag unset the venv runs stock code, as shown above.

### 4x. Cold PLE pages: readahead beats waiting (follow-up to #58439; `tools/plecold/willbench.py`, `patch_fill.py`)
**Idle box**, 57 cold pages per rep, page cache dropped, 25 reps (`data/plefill/willbench*.json|txt`):

| fill | median |
|---|---|
| serial touch (the #58439 path for < 4096 rows) | 4.6–4.7 ms |
| C helper, 16 pthreads × `MADV_POPULATE_READ` (§4h) | 0.69–0.83 ms |
| `MADV_WILLNEED` for every page, then a touch | 0.36–0.41 ms |
| `MADV_WILLNEED` for every page, then `MADV_POPULATE_READ` per page (ctypes, GIL-free) | **0.355 ms** |
| issuing `MADV_WILLNEED` alone | 0.11 ms |

**In the server** (`plefill`; clone venv, RecoverSSM off, KV 4 GiB, `coldprobe.py` cold pass + warm pass of 6
requests; 2 starts per arm, alternating):
- arms: base; auto (the §4j gated wait + C helper); will (the same, with the fill replaced by WILLNEED +
  POPULATE_READ).

| ms/step | base | auto | will |
|---|---|---|---|
| cold pass, start 0 / 1 | 59.15 / 59.62 | 57.34 / 57.06 | **56.98 / 56.67** |
| warm pass, start 0 / 1 | 54.61 / 54.78 | 54.87 / 54.95 | 54.91 / 54.83 |

- **will vs auto:** faster on 12/12 paired cold requests (−0.14…−0.60 ms). **will vs base:** −1.15…−4.12 ms.
  Hashes are identical to base in all 72 requests.
- **The mechanism is not the wait.** Readahead turns the step's major faults into minor ones, so will's fault EMA
  fell to ~0 and the auto gate stopped waiting after ~13 steps. The rest of the run is readahead with no wait: the
  GPU gather still takes ~1.8–2.0 ms (base 3.7–5.5), because its faults land on reads already in flight in
  parallel, and the host keeps its one-step lead.
- **Consequence for the follow-up PR:** no wait, no fault-rate gate, no config key and no compiled helper. The fill
  replaces the serial touch for decode-sized row sets (+94/−1 lines). Hypothesis H1 held, for a different reason
  than written.

### 4y. Deferred commit: kernels pass (`tools/rssm/defer/`)
`test_defer.py` runs 7 steps × 3 requests with random acceptance, comparing the immediate and the deferred path of
the same kernel file. Mode `align` uses block size 6, so every request crosses 3 boundaries.
- Every block state (boundaries plus the final running state) agrees to 1.05e-7 (`none`) and 8.8e-8 (`align`)
  relative. Verify outputs agree to 3e-4 (bf16).
- A stale pending count on a reused block is cleared.
- **Design as built:**
  - a row that crosses no block boundary records its accepted count in a per-block `pending` counter;
  - the next verify replays those records forward, stores the checkpoint, then verifies;
  - boundary rows commit immediately, and clear source and target;
  - prefill rows clear their block's counter in the builder;
  - k/g are stored per value tile, because the next verify rewrites the record while sibling tiles still read it.
- Data: `data/rssm/defertest-0926.txt`. Server A/B next.

### 4z. The #58439 follow-up, as it will be posted: readahead fill, no wait — cold −2.35 / −2.91 ms/step
- **PR code** (`jschmied:pr/ple-cold-fill`, commit `3f2142c111`): `MappedTable.touch` fills decode-sized row sets
  (< 4096 rows) with `MADV_WILLNEED` for every page, then `MADV_POPULATE_READ` per page, both via ctypes. There
  is no wait, no option and no compiled helper; before Linux 5.14 it falls back to the plain touch.
- **Unit tests** on the exact branch base (a venv from the main `378504a54` wheel + the #58439 head files;
  `vllm-venv-plepr`): 30 passed. The 2 new tests fail on the #58439 head. Data: `data/plefill/pleprtest-*.txt`.
- **Current main cannot load our local checkpoints:** the new `MergedColumnParallelLinear.load_weights` falls back
  to the module for keys it does not know. So the serving A/B replays the PR path on the prod-based stack
  (`patch_fill2.py`, `FN_PLE_FILL=nowait`, the same page arithmetic and calls) — `plefill2b`, 2 starts per arm:

| ms/step | base | PR path |
|---|---|---|
| cold pass, start 0 / 1 | 59.15 / 59.41 | **56.80 / 56.50** |
| warm pass, start 0 / 1 | 54.73 / 54.71 | 54.95 / 54.75 |

- 12/12 paired cold requests are faster (−1.37 … −3.78). Major faults per step fall from 29–33 to 0.1. Hashes
  are identical in 48/48 requests. All three pre-written hypotheses held (`tools/plecold/HYPOTHESIS.md`).

## Step 5 — round 2 (2026-09-26 afternoon, "continue with lightspeed agenda")

### 5a. Deferred commit (§4w/4y), server A/B: null at c=1, slightly worse at c=4 — not a promotion candidate
`rssmdefer`: clone venv + `tools/rssm/defer/patch_defer.py`; arms immediate vs deferred (`FN_GDN_RECOVERSSM_DEFER=1`),
align + prefix caching, MTP n=3, KV 4 GiB, `comboprobe2`, 2 starts. Path lines `deferred False|True` required.
Data: `data/rssm/rssmdefer*`.

| | immediate | deferred |
|---|---|---|
| c=1 ms/tok | 21.496–21.549 | 20.913–20.949 |
| c=1 accept length | 2.53 | 2.59 |
| **c=1 ms per verify cycle** | 54.38–54.52 | **54.16–54.26** (−0.2…−0.5 %) |
| **c=4 ms per verify cycle** | 100.29 (start 1; start 0 see below) | **100.84–101.03** (+0.5 %) |
| agent loop, s/turn (prefix hit rate) | 1.11 (82.1 %) | 0.93–0.94 (89.1 %) |

- **The per-cycle times are the fair speed metric, and they are null.** The ms/tok, acceptance and agent-loop
  differences come from a different text trajectory:
  - deferral changes the GDN state at summation-order size (first divergence vs immediate at a median of 31
    tokens);
  - so the drafts, the acceptance (2.53 → 2.59), the agent conversation and its prefix-hit pattern (82 → 89 %)
    all differ;
  - nothing separates that from a real gain, so none of it is attributed to the deferral.
- Against the hypothesis: c=1 at the bottom edge of −0.5…−1.3 %; **c=4 outside** (expected −1…−2.5 %, got
  +0.5 %). The forward replay inside the latency-bound verify kernel (§4v) costs about what the saved state read
  gave back.
- **Correct:** cache-hit replay 8/8 in both deferred starts; each arm reproduces itself across restarts; kernel test
  §4y. The KV pool shrinks 0.8 % (larger replay record).
- **Verdict: not promoted, not pursued.** It is ~0.4 % at c=1 for extra state across steps and a block-handover
  hazard. The code stays env-gated in the clone and in `tools/rssm/defer/`.
- **Anomaly, recorded:** in immediate start 0, the second (cache-hit) c=4 pass diverged in all 4 requests and ran
  89.45 tok/s, while its first c=4 pass matched every earlier run. Start 1 of the same arm and code reproduced the
  §4t hashes in both passes (99.31 tok/s). There was no preemption, no error, and the same 82.1 % hit rate. This
  reads as batch-composition drift at c=4 (greedy is not batch-invariant under concurrency), not corruption; the
  counterfactual is the clean start 1 on identical code.

### 5b. Dynamic draft stop on the NVFP4 draft head (FNDYN2): rejected — +5…6 % slower at c=1
`fndyn2`: FNDYN (finding 235: stop drafting once every request's running confidence product < 0.3), with the
confidence now also computed on the FNNVFP4 head's float32 logits (`tools/dyndraft/fndyn2_patch.py`). Clone venv,
prod-like (RecoverSSM, NVFP4 head, 32k vocab, MTP n=3), KV 4 GiB, `comboprobe2`, 2 starts. Data: `data/dyn2/`.

| | base | FNDYN2, thr 0.3 |
|---|---|---|
| c=1 ms/tok | 21.474–21.549 | **22.599–22.851** |
| c=4 tok/s | 99.83–99.96 | 96.13–96.60 |
| agent loop s/turn | 1.10–1.11 | 1.12 |
| early stops | — | 13.6 % of cycles (stop before step 1: 3 %, before step 2: 10 %) |

- **Outside the hypothesis on both counts:**
  - Speed: expected −0.5…−1.5 % at c=1, got **+5…6 %**. The stop check costs one host sync per draft step, and with
    the NVFP4 head a draft step is now cheaper than that sync. On the BF16 head (finding 235) the balance was the
    other way, −1.7 %.
  - Correctness: expected hash-identical, but **c=1 text differs from base**. It is reproducible within the arm;
    first divergence at a median of 186 tokens, |Δlogprob| 0.002 before it. c=4 hashes were identical (the stop
    rarely fires there).
- **Why the text moves (a RecoverSSM property, not an FNDYN bug):** the committed state depends on how accepted tokens
  are grouped into commits (fp32 rounding of the replay). A different acceptance pattern (repeated-token drafts get
  rejected) gives a slightly different state, and greedy text diverges late. Native spec decode stores a state per
  position and does not have this property. Size: smaller than a reduction-order change (median 186 vs 26–31
  tokens).
- **Verdict:** rejected; FNDYN stays out of prod. The patch stays in `tools/dyndraft/` (env-gated, off).

### 5c. QSA indexer under speculative decode: rejected drafts cannot leak (code audit + GPU check)
**Question** (from the user's state-category analysis): the QSA indexer keeps a raw-key ring and compressed 4-token
rows outside the main KV cache. Can a rejected draft's key be finalized into a compressed row, or linger in the
ring, where an accepted token later reads it? This is correctness, not speed: the main KV is slot-addressed, and the
GDN/PLE conv states already use accepted-suffix compaction (~50 µs/step, §4u).

**Code audit** (read-only; paths under `models/qwen4_exp/`):
- **Raw-key ring: SAFE.**
  - It holds 4·⌈(4+n)/4⌉ = 8 rows at n=3, sized so a rejected draft cannot overwrite a committed key
    (`common/qsa_cache.py:839-856`). The slot is `pos % 8`.
  - Keys inside the current step come from the step's fresh keys; the ring is read only for positions before the
    step's first token (`nvidia/ops/qsa_pre_indexer.py:236-260`, unfused `ops/qsa.py`).
  - Writes follow the reads (`qsa_pre_indexer.py:346-372`). A rejected draft sits at most 5 positions ahead, under
    the ring width.
- **Compressed rows: SAFE.**
  - A query at q selects only `visible = min((q+1)//4, seq_len//4)` rows, i.e. groups with 4g+3 ≤ q
    (`qsa_cache.py:268-275`).
  - The open group is served from the main KV (`nvidia/ops/qsa_indexer.py:238-262`).
  - Row g is written only when 4g+3 is in the batch, to slot `pos//4`, and rewritten before any later reader.
- **Other state: SAFE.** `rope_position_cache` follows the ring; `first_positions` and `k_work_metadata` are rebuilt
  every forward. The drafter's top-k reuse can only lower acceptance.

**The one assumption, checked on the GPU** (`tools/qsa/qsa_visible_check.py`, `data/qsa/qsa_visible_check-0926.txt`):
- The audit relies on top-k never returning a block index ≥ the row's `visible`: those logit columns hold leftovers.
- The check runs the server's own `_topk` and the full `qsa_select_paged_decode` path:
  - stock `_C.persistent_topk` and the deterministic `_C_det` variant;
  - visible = [g, g, g, g+1] for g ∈ {1, 5, 511, 512, 2047};
  - k ∈ {512, 2048};
  - widths 4096 and 65536;
  - columns ≥ visible pre-filled with +inf, then 1e30.
- **160/160 pass, 0 fail.**
- (k=16 cases error by design: the op supports k ∈ {512, 1024, 2048}; prod uses 512.)

**Verdict:** no transactional handling is needed in the QSA indexer; it is already correct under MTP.

### 5d. CUDA-graph capture widths at concurrency: ≤ 2 % — decoding c ≥ 3 without graphs costs little here
With MTP n=3 a decode row is 4 tokens, so the prod list `[1,2,4,8]` covers c ≤ 2 only; c ≥ 3 runs the target
without graphs. (bilikaz's recipe captures multiples of K+1 up to seats×(K+1); MiaAI's kit covers every width.)
`cgwidth`: `[1,2,4,8]` vs `[1,2,4,8,12,16,24,32,48,64]`, prod-like (RecoverSSM, NVFP4 head, 32k vocab), KV 4 GiB,
`decprobe` (c=1, c=4, agent) + `tools/cg/concprobe.py` (c = 4/8/16, distinct prompts, 400 tokens), 2 starts.
Data: `data/cg/`.

| | `[1,2,4,8]` | `[…,64]` | comparable? |
|---|---|---|---|
| decprobe c=4 tok/s | 99.74 (start 1; start 0 drifted to other text: 89.57) | 101.85–101.98 | yes, identical hashes → **+2.1 %** |
| concprobe c=4 / c=8 / c=16 tok/s | 87.7–88.7 / 127.2–128.7 / 161.1–167.1 | 88.7–89.0 / 128.7–131.4 / 165.0–166.0 | overlapping; texts differ at c=8/16 → null within noise |
| c=1 ms/tok (accept len) | 21.49–21.53 (2.53) | 21.06–21.10 (2.586) | text differs; per cycle 54.4 vs 54.5 ms → equal |
| graph memory | 0.31 GiB | 0.56–0.59 GiB | KV pool unchanged (103,953 tokens) |

- **Outside the hypothesis** (+5…15 % at c=4): the step is GPU-bound at these widths, and async scheduling hides
  the per-kernel launch cost of the graph-less forward.
- **Prod:** +~2 % at c=4 for 0.28 GiB of graphs is cheap but small; it is the user's call. It does not matter for
  single-stream agent work.
- **Measurement lesson, again:** c=4 greedy text is not stable run to run (start 0 of the baseline drifted, like
  §5a's). Concurrency numbers are compared only where the hashes match.

### 5e. `vm.compaction_proactiveness` 20 → 0: null at our memory headroom
Source: bilikaz's recipe (a 4–5 s stall every ~37 s, ~10 %, at their 27 GB KV with ~3 GB free). `compact`: the
launcher wrapper sets the sysctl per arm, and the driver restores 20. Prod-like, KV 4 GiB. Probe: decprobe (one pass,
not warmed, so its c=1 ms/tok is not comparable with other sections), then 4 streams × 3,500 tokens sustained,
timed in 5 s windows. 2 starts. Data: `data/compact/`, tools `tools/compact/`.

| | 20 (default) | 0 |
|---|---|---|
| sustained c=4, chunks/s mean (per start) | 36.38 / 36.92 | 36.83 / 35.84 |
| windows < 70 % of median | 0 / 0 | 0 / 0 |
| p10 window | 36.0 / 36.0 | 36.0 / 35.2 |
| kcompactd pages scanned during the run | 0 / 16.0 M | 2,596 / 0 |

- **Null:** the compactor ran in one default start (16 M pages scanned) with no dip and no throughput cost. Outputs
  are identical (c=1 hashes = §4t's). Hypothesis H1 (+2…10 %) missed; the "outside" line applies: their stall needs
  their memory edge.
- **Not measured:** prod's default KV (~33 GiB, closer to their edge) under sustained load. If prod ever runs that
  close to the edge, re-test there before dismissing it. Not a change for prod now.

### 5f. Semantic work audit: is each kernel family doing only the work the model needs? (read-only, 5 parallel audits)

The user's question: GEMMs at their byte floor can still be doing avoidable work, such as padded tiles, weights read
for idle experts, logits nobody reads, or copies that only change layout. Five read-only audits used the §4u node
trace (`rssm.sqlite`, 48 verify cycles, 54.23 ms), the installed code and the notes. Scripts are in
`tools/audit0926/`. Proof types: **A** trace-measured, **B** code + byte arithmetic, **C** estimate that needs a
counterfactual run.

**Load-bearing table, kernel families > 0.2 ms/step**

| family | ms/step | verdict | avoidable | proof |
|---|---|---|---|---|
| routed MoE GEMM1+GEMM2 (NVFP4 grouped) | 17.5 | required. Tile padding (851/1,702 physical rows vs 40 useful pairs) costs FLOPs, not time: each active expert's weights stream once. Zero-row experts get a 0-size problem (`cutlass_fused_moe_kernels.cuh:1392-1405`), so no weights are read; 40 reads/call would need 318 GB/s | ~3–5 % | A+B |
| MoE block total (router → finalize) | ~21.7 | 1.09× byte floor (425 vs 391 µs/call) → ceiling ~1.7 ms/step | ≤ 1.7 ms | A+B |
| shared expert (BF16, 4 kernels) | 4.49 | required and **hidden**: side stream, 57 of its 69 µs GEMM time falls inside the router+prologue window. Joining the grouped GEMM would need NVFP4 shared weights and expert 513 / top-11; ≈ −0.5 ms at best | ~0 critical-path | B |
| BF16 8x10 "slow site" (§4a/4l) | 2.25 | **solved:** it is the shared expert's gate_up, contending with the concurrent router GEMM for DRAM (201 GB/s combined, the standalone rate). Off the critical path | 0 | B |
| HC mixers down/up (BF16, 106 sites) | 6.96 + glue 0.8 | required. No static linear→linear chain (rrms, silu, σ between every pair), so nothing folds offline. Tail fusion (reduce+silu prologue, σ+gate_mix epilogue) saves launches only; 12 zero pad rows in W_down ≈ 0.12 ms | ~0.5–0.6 ms | B |
| FP8 dense (qkvz, out/o_proj) | ~10.2 | required (at floor, §2) | ~0 | A |
| lm_head (FP8, 248k rows) | 2.68 | weight read required. Post-GEMM logit work is only 53 µs/step (greedy); a fused argmax epilogue would save ≤ 35 µs; top-20 for sampled is exact in distribution but not bit-identical | ≤ 0.07 % greedy, ≤ 0.3 % sampled (unmeasured) | A+B |
| MTP drafter (3 steps) | 5.07 | load-bearing at c=1. The third draft pays 14.3 ms per accepted token vs a 22.0 average. **But** its BF16 dense qkv/o/input projections (~2 ms/cycle) are not load-bearing at BF16 | ~1.4 ms (NVFP4) | A+B |
| extra verify rows (MoE) | ~3.3/row | **new:** verify does *not* grow little with M. Each extra row adds ~4.9 distinct experts × 48 layers, about 2× a draft step. Two-thirds of a draft position's marginal cost sits here | — | A+B |
| GDN eager glue (q/k/v copies, CatArray, out memcpy) | 0.74 incl. gaps | **excessive:** the verify wrapper accepts strided views and has `out=`, which `_forward_core` does not use | ~95 % | B |
| GDN graph glue (b/a/z `.contiguous()`, `zeros`) | 0.25 | excessive / wasted at uniform M=4 | ~80 % | B |
| finalizeMoeRouting (4 CTAs) + shared add | 0.50 | required math, under-parallel (one CTA per token) | ~70 % | C |
| target input prep (103 eager ops) | 0.46 | bookkeeping (`mamba_get_block_table_tensor` pattern ×4 groups) | ~65 % | A/B |
| drafter steps 2–3 ungraphed | 0.30 | launch overhead (PIECEWISE forces NONE for draft decodes) | ~80 % | A |
| FP8 act-quant ×96 | 0.23 | bookkeeping, fusable into the producers' epilogues | ~100 % | B |
| QSA (15 calls, eager) | 0.58 + 0.39 gaps | required math. At short context, 89–97 % of 64 split-K CTAs are empty but each still writes 6 MiB of fp32 partials per layer; at 32k the 4 verify rows gather ~70 % duplicate tokens | 20 % short / ~70 % of gather at 32k | C |
| RecoverSSM commit | 1.11 | required, at DRAM floor (§4u) | ~0 | A |

**Answers to the seven questions:** (1) padding is FLOPs, not time; (2) no weights are read for zero-row experts
(code + traffic proof); (3) HC has no foldable chain, only launch fusion (~0.9 %); (4) the shared expert is already
free on the critical path, so merging it is ≈ −0.5 ms at best, at quality risk; (5) k=3 is right at c=1, k=2 is
marginally better at c=4 (−1.4 % ± 2, untested), and the cheap lever is the drafter's BF16 dense layers; (6) the
logits are not the lm_head cost, the weights are; (7) QSA wastes CTAs and partial traffic at short context and
duplicate gathers at 32k, both small at our probe lengths.

**Summary:** no big hidden waste. The big families do required work at their floors. Removable work adds up to
about 4–5 ms/step of small items, and the gap share may not transfer 1:1 (finding 237: FULL graphs gained only
−0.7 %).

**Queued counterfactuals, cheapest first** (hypothesis written per run):
1. **GDN views + `out=`** (Python only, same kernel on the same values): −0.3…−0.8 ms/cycle, hashes must stay
   identical. This also tests how much of the eager-gap time really goes away.
2. **MTP layer dense → NVFP4** (reuse `_nvfp4_rows_gemv_kernel`, finding 234 recipe): −2.6 % c=1 predicted.
   Compare per-cycle time, since RecoverSSM text moves with acceptance (§5b).
3. **n=2 vs n=3 at c=4** on the current stack (predicted −1.4 % ± 2 in favour of n=2).
4. HC tail fusion on FNBF16SK (−0.5 ms), dropping the HC pad rows (−0.12 ms), 2-D finalize / fused-finalize
   tactic (−0.3…0.45 ms, determinism gate), and `in_proj_ba` 8x1 re-test (7.8× its floor; the FNBF16SK null predates
   RecoverSSM).

### 5g. A1 — GDN verify on strided views + `out=` (FNQKVVIEW): −1.2…−1.6 % c=1, outputs identical

The first §5f counterfactual. Under RecoverSSM verify, `_forward_core` handed `gdn_recoverssm_verify` contiguous
copies of q/k/v (one `CatArrayBatchedCopy` + three `direct_copy` per GDN layer) and then copied its output into
`core_attn_out` (a 49,152 B D2D memcpy). The kernel already takes token-strided heads and has an `out=` parameter.
Patch (`tools/qkvview/patch_qkvview.py`, env `FNQKVVIEW=1`, clone venv only) passes views of the conv output and
writes in place: 180 launches/step and ~12 MiB/step of copies gone, same kernel on the same values. Unit check
(`test_views.py`, GPU): output and replay record bit-identical at batch 1/2/4. Hypothesis (written before):
−0.3…−0.8 ms/cycle, hashes identical. Run `qkvview1`: 2 starts per arm, alternating, KV 4 GiB. Data:
`data/qkvview/`.

| | view (FNQKVVIEW=1) | base |
|---|---|---|
| c=1 ms/tok (per start) | 21.293 / 21.285 | 21.544 / 21.644 |
| c=1 per verify cycle (× 2.53) | 53.87 / 53.85 | 54.51 / 54.76 |
| c=1 hashes | = reference (`2f3574ed…`) both | = reference both |
| c=4 tok/s, hash-matched cells | 100.13 | 99.68 / 99.37 |
| cache replay | 8/8 in view1; view0's c=4 pass drifted (below) | 8/8 both |
| agent s/turn | 1.10 / 1.10 | 1.11 / 1.11 |

- **Win: −0.64…−0.91 ms per cycle at c=1 (−1.2…−1.6 %)**, identical text. It lands at the top of the predicted range,
  so most of the eager-gap time does go away along with the kernels. That answers §5f's open question for this site:
  unlike finding 237's FULL-graph arm, removing dependent eager ops does transfer here.
- **c=4:** view0's measured c=4 pass drifted to different text on all 4 prompts (accept 2.47, 90.2 tok/s), while its
  first pass matched the reference. That is the known c=4 run-to-run instability (§5d note: 2 of ~10 earlier passes
  drifted, base-only). The same kernel on the same values is bit-identical in the unit check. Excluded by the hash
  rule; the matched cells give +0.5…+0.8 %.
- Next: fold it into the RecoverSSM review branch (it belongs to the verify path) and into the prod candidate.
  The graph-side b/a/z copies and `zeros` (§5f item 6, ~0.25 ms) are the natural follow-up.

### 5h. A2 — MTP drafter dense layers to NVFP4: rejected, acceptance drops more than the step saves

§5f's second counterfactual. The drafter's fc_embedding, fc_hidden, self_attn.qkv_proj and self_attn.o_proj are BF16
(~125 MB read per draft step). Patch `tools/mtpd4/` (`FN_MTP_DENSE_NVFP4=1`, clone venv): NVFP4 copies in the
draft-head layout through the draft head's rows GEMV as a custom op (`fn_nvfp4_linear`). Microbench (L2 flushed,
graph replay): 709 → 231 µs per single-row draft step, 565 → 223 µs at M=4, i.e. ~−1.3 ms per verify cycle; weight
error ~9.5 % relative (random-weight proxy). Hypothesis (written before): −0.9…−1.4 ms/cycle, accept_len within
−0.05…+0.03; H0 if accept_len drops > 0.08. Run `mtpd4a`, both arms `VLLM_DISABLE_COMPILE_CACHE=1`; stopped after one
start per arm (below). Data: `data/mtpd4/`.

| | NVFP4 dense | base |
|---|---|---|
| c=1 ms/tok | 21.725 | 21.237 |
| c=1 accept_len | 2.443 | 2.53 |
| c=1 per cycle | 53.07 ms | 53.73 ms |
| c=4 tok/s (accept) | 97.66 (2.413) | 100.39 (2.49) |
| agent s/turn | 1.09 | 1.10 |

- **H0:** the cycle got 0.66 ms cheaper (a bit under the predicted −0.9), but acceptance fell 0.087, so every token
  costs **+2.3 %** at c=1 and c=4 loses 2.7 %. Unlike the experts (finding 198) and the sliced head (finding 234), the
  drafter's attention/input projections do not tolerate NVFP4 with plain per-16 max scaling.
- One start per arm only: the acceptance drop is 0.087 on the same prompts where base reproduces 2.53 in every run
  today (rssmr2d, qkvview1, rssmr3a/b), so a second start would not change the verdict. Stopped early to run the FP8
  variant (user: "fp8 drafter then instead of fp16"), §5i.

### 5i. A2b — the same four drafter layers as FP8 (per-row scales): also rejected

The user's follow-up ("fp8 drafter then instead of fp16"). Same patch, `FN_MTP_DENSE_FP8=1`: FP8 E4M3 with one FP32
scale per output row through a Triton rows GEMV (`tools/mtpd4/fn_nvfp4_dense.py`), weight error ~2.6 % (vs ~9.5 % for
NVFP4). Microbench: 698 → 311 µs per single-row draft step, 568 → 308 µs at M=4, ~−1.03 ms per cycle. Hypothesis
(`tools/mtpd4/HYPOTHESIS-fp8.md`): accept_len within −0.03, −1.3…−2.0 % ms/tok; H0 if accept_len drops > 0.05. Run
`mtpd8a`, one start per arm (stopped, same reasoning as §5h). Data: `data/mtpd4/mtpd8a.*`.

| | FP8 dense | base |
|---|---|---|
| c=1 ms/tok | 21.492 | 21.308 |
| c=1 accept_len | 2.468 | 2.53 |
| c=1 per cycle | 53.04 ms | 53.91 ms |

- **H0 again:** the cycle is 0.87 ms cheaper, but acceptance drops 0.062, so each token costs **+0.9 %**. (c=4 is not
  usable here: base0's c=4 pass drifted to different text, accept 2.446.)
- So the drafter's dense projections are precision-sensitive even at FP8 per-row, while its routed experts and the
  sliced head tolerate NVFP4. A plausible reason: the MTP layer's attention and input projections feed a single
  layer whose argmax must agree with the target's, with no later layers to average the error out. A per-layer probe
  (`FN_MTP_DENSE_LAYERS`) could find a tolerant subset, but the best case is ~1 ms/cycle, so it is parked behind the
  GDN-projection NVFP4 test (§5j), which targets ~4 ms/step.

### 5j. GDN projections as NVFP4 W4A16 (Marlin): −5.8 % c=1, +4.5 % c=4, at +0.4…0.6 % NLL (quality trade, the user's call)

The user's request after the myllmbox hibrid48 review ("ok, try it"). The 36 GDN layers' `in_proj_qkv` + `in_proj_z`
(fused `in_proj_qkvz`) and `out_proj` run FP8 block today. myllmbox ships them as NVFP4 W4A16 through Marlin.
- **Source weights:** the BF16 originals came from the backup box, from RadixArk's NVFP4 checkpoint, whose ModelOpt run
  ignored `*.linear_attn.*`. Only the 108 tensors were extracted (4.15 GB, per-tensor sha256 in
  `data/gdn4/bf16-extract-SOURCE.json`, `tools/gdn4/extract_gdn.py`). Our FP8 copies differ from them by exactly the FP8
  rounding (2.6 % relative), so they are the same weights.
- **Variants** (`tools/gdn4/build_gdn_variants.py`): `mtpfp4-gdn4` (NVFP4, codes rounded against the effective
  fp8-rounded scale, qkv and z share one global scale because vLLM fuses them; weight error 9.36 %) and
  `mtpfp4-gdnbf16` (the BF16 reference). Everything else is hardlinked; 4 shards rewritten without the FP8 GDN tensors.
  vLLM loads it natively (`quantized_layers: W4A16_NVFP4`, log: `Using MarlinNvFp4LinearKernel`).
- **Microbench** (`tools/gdn4/bench_marlin.py`): Marlin NVFP4 runs at its byte floor on GB10 (qkvz 106.8 µs vs a 107.2
  floor; out_proj 42.2 vs 40.2), against FP8 floors of 190.7 / 71.5 µs.

**Speed** (`gdns`, 2 starts per arm, RecoverSSM + align + prefix cache, KV 4 GiB; data `data/gdn4/gdns.*`):

| | NVFP4 GDN | FP8 GDN (current) |
|---|---|---|
| c=1 ms/tok | 20.050 / 20.061 (≈49.9 tok/s) | 21.299 / 21.271 (≈46.9 tok/s) |
| c=1 accept_len | 2.492 | 2.53 |
| c=1 per cycle | 49.97 ms | 53.85 ms (−3.9 ms, −7.2 %) |
| c=4 tok/s | 104.57 / 105.02 | 100.50 / 100.33 |
| agent s/turn | 1.05 / 1.04 | 1.10 / 1.10 |
| cache replay | 8/8 both starts | 8/8 both starts |

Within the hypothesis (`tools/gdn4/HYPOTHESIS.md`: −4.6…−7.4 % per cycle): the ~3.9 ms/step byte saving arrives
in-model almost completely. Acceptance moves (2.53 → 2.49) because the target's outputs change; the text is different
from the FP8 arm (expected, not a defect).

**Quality** (`gdnq`, teacher-forced prompt logprobs over 3 long documents, 25,885 tokens; reference = BF16 GDN
projections; `tools/gdn4/lpdist.py`, output `data/gdn4/lpdist.txt`):

| GDN projections vs BF16 | median \|Δlp\| | p90 | tokens with \|Δlp\| > 0.5 | NLL |
|---|---|---|---|---|
| FP8 (current) | 0.074 | 0.68 | 14.7 % | −0.19 % |
| NVFP4 | 0.086 | 0.77 | 17.6 % | **+0.40 %** (docs +0.04 / −0.20 / +1.51 %) |

- On these documents any GDN perturbation already produces most of the per-token spread (the fp16-state test in §4r
  shows the same size, while the identical computation gives exactly 0). So the |Δlp| distribution barely separates
  NVFP4 from FP8 (+15 %). NLL is the clearer signal: **+0.40 % vs BF16, +0.58 % vs our FP8**, with one document at +1.5 %.
- FP8 GDN is effectively free (−0.19 % vs BF16).
- **Verdict: a quality trade, not free speed**, the same class as §4r. −5.8 % c=1 / +4.5 % c=4 / −5 % agent turn for
  ~0.4–0.6 % NLL. No task-level eval yet (SWE-bench resolves would decide). Not promoted; the user decides.

### 5k. Draft vocabulary: our 32k slice vs TensorFold's 79,591 ids — coverage gap 0.1–0.6 pp, no A/B

From the TensorFold review (ashhart/TensorFold, a standalone CUDA engine for this model; its 79,591-id draft list beat
a 98,755-id list by 7.7 % on sampled code there). Offline, CPU only: the share of tokens of real text that fall inside
each list (the model's tokenizer; the three gdncs documents plus ~65k tokens of Python and CUDA source).

| text | tokens | in our 32k slice | in their 79.6k list |
|---|---|---|---|
| doc a / b / c | 14,643 / 4,979 / 12,652 | 99.31 / 99.04 / 99.32 % | 99.88 / 99.68 / 99.76 % |
| Python source | 53,977 | 99.81 % | 99.97 % |
| CUDA source | 11,669 | 99.86 % | 99.86 % |

- The lists share 31,686 ids; ours has 1,082 they lack, theirs 47,905 we lack.
- A draft can only be lost to the slice on the 0.1–0.6 % of tokens outside it. Growing the head 2.4× costs time on every
  draft step (§4 finding 234: the slice size is the head's bytes), so this is a **null: keep 32k**.

### 5l. MTP depth on code vs prose: K=5 −10 % on code (65.8 tok/s), +4…6 % on prose — the text decides K

Question from comparing with bilikaz's recipe (K=5, 5.06 accepted per step on a code prompt). Our speed probes are
prose ("Explain in about 500 words…"). New probe `tools/ksweep/codeprobe.py`: 4 code-writing prompts (Python, TypeScript,
Rust, C++) and decprobe's 4 prose prompts, 700 tokens, thinking off; cells: code c=1 greedy, code c=1 sampled (the
model's defaults 1.0/0.95/20, fixed seeds), prose c=1 greedy, code c=4 greedy. Each prompt is seen once per server, so
the numbers include cold PLE pages (≈7 % above decprobe's warm second pass; compare across K only). Stack: RecoverSSM +
align + prefix cache, FP8 GDN, capture sizes [1,2,4,5,6,8,10,12,16,20,24] in every arm; K=5 needs `--block-size 1728`
(QSA ring capacity 12 must divide it; the auto size is 1696 here, so bilikaz's 1632 does not fit). 2 starts per K.
Hypothesis: `tools/ksweep/HYPOTHESIS.md`. Data: `data/ksweep/`.

| cell (ms/tok; c=4: tok/s) | K=3 | K=4 | K=5 |
|---|---|---|---|
| code c=1 greedy | 16.90 / 16.99 (acc 3.36) | 16.11 / 16.09 (3.81) | **15.18 / 15.21 (4.35)** |
| code c=1 sampled | 17.87 / 18.00 (3.15) | 17.79 / 17.83 (3.42) | **17.15 / 16.90 (3.89)** |
| prose c=1 greedy | **23.15 / 23.32 (2.53)** | 23.65 / 23.74 (2.67) | 24.46 / 23.98 (2.83) |
| code c=4 | 133.4 / 131.3 | 132.8 / 141.8 | **145.4 / 148.3 (4.48)** |

- **Code: K=5 wins** (−10.3 % vs K=3 greedy, −4.7 % sampled, +10 % at c=4), inside the hypothesis (5–12 %). The
  step is ~66 ms at 4.35 tokens (bilikaz: ~69 ms at 5.06 with sampled drafts).
- **Prose: K=3 wins**, K=5 is +3…6 % slower. Per-position acceptance on prose falls to 0.19 / 0.12 at positions 4/5.
- **The greedy→sampled gap grows with K:** 0.21 / 0.39 / 0.46 accepted tokens per step at K=3/4/5. That is the headroom
  probabilistic drafting (queued) would target.
- Greedy c=1 hashes are identical across starts in every cell; c=4 hashes differ between starts at K=3/4 (the known c=4
  run-to-run instability), so the c=4 row is indicative only.
- **Consequence:** the best static K depends on the traffic. For code-heavy agent work K=5 (with block 1728) is the better
  default; for chat/prose K=3. A per-request or adaptive depth (vLLM's `enable_adaptive_verification` is DSpark-only
  today) would get both. Not a prod change without the user.

### 5m. F1 — PIECEWISE graphs for MTP draft steps 2..K: null (−1.6 % prose at K=3 only), not carried

First item of the fusion plan (user: "sounds like a plan"). Under PIECEWISE, vLLM runs draft decode steps 2..K without
graphs ("PIECEWISE cudagraphs are not supported for draft decodes" → NONE). `tools/f1/patch_draftpw.py`
(`FN_DRAFT_PW=1`) keeps PIECEWISE for them; the draft-step function already takes any runtime mode. Run `f1dpw`, the
code/prose probe of §5l, K=3 and K=5 (block 1728), 2 starts per arm, stack = the PR-equivalent code (`--use-replayssm`).
Hypothesis (`tools/f1/HYPOTHESIS.md`): −0.3…−0.6 % at K=3, −0.5…−1.0 % at K=5, hashes identical. Data: `data/f1/`.

| ms/tok | K=3 base | K=3 graphed | K=5 base | K=5 graphed |
|---|---|---|---|---|
| code greedy | 16.95 / 16.74 | 16.84 / 16.83 | 15.20 / 15.06 | 15.26 / 15.30 |
| code sampled | 18.00 / 18.10 | 17.92 / 17.87 | 16.95 / 16.91 | 17.10 / 16.95 |
| prose greedy | 23.54 / 23.42 | **23.08 / 23.10** | 25.28 / 24.28 | 24.52 / 24.43 |

- Output hashes are identical between arms in every c=1 cell, so the patch is exact.
- Only K=3 prose separates (−1.6 %); code overlaps; at K=5 the graphed arm is if anything slower on code (+0.9 %).
  The hypothesis expected the K=5 gain to be the larger one — it is not there. So the eager draft steps are not the
  cost the audit's gap count suggested (host launch is hidden by async scheduling; the drafter's compiled region per
  step is small). Not carried; the patch stays env-gated in the clone venv. F4 (FULL graphs) would still graph these
  steps as a side effect, but this result lowers its expected value.

### 5n. Probabilistic drafting over the 32k NVFP4 draft slice: −5.5 % on sampled code at K=5, exact, free on greedy

Queued at the user's request ("put probabilistic draft sampling for our head into queue"), sized by §5l's
greedy→sampled acceptance gap (0.21 @K3, 0.46 @K5). vLLM's `draft_sample_method='probabilistic'` samples each draft
from q = softmax(draft logits / T) and runs the exact ratio test with the same q. Our drafter used local argmax over a
32k NVFP4 slice of the head, which vLLM refuses together with probabilistic drafting. Patch `tools/dprob/patch_dprob.py`
(`FN_DRAFT_PROB=1`): in probabilistic mode `Qwen4ExpMTP.compute_logits` returns the slice logits scattered into a
persistent full-vocab buffer filled with −inf once (DSpark's reduced-vocab mechanism), so q = 0 outside the slice and
the test stays exact. Launcher copy with a `draft_sample_method` knob (`tools/dprob/serve-dprob.launcher.diff`; prod's
launcher untouched). Run `dprob`: codeprobe (§5l), K=3 and K=5 (block 1728), greedy drafts (local argmax, base) vs
probabilistic, 2 starts. Hypothesis `tools/dprob/HYPOTHESIS.md`. Data `data/dprob/`.

| | K=3 greedy drafts | K=3 probabilistic | K=5 greedy drafts | K=5 probabilistic |
|---|---|---|---|---|
| code sampled, ms/tok | 18.01 / 17.98 | 17.87 / 17.98 | 16.89 / 17.04 | **16.07 / 15.92** |
| code sampled, accept_len | 3.152 | 3.186 | 3.894 | **4.171** |
| code greedy, ms/tok (hash) | 16.96 / 16.99 (e10d01) | 16.86 / 17.06 (e10d01) | 15.02 / 15.33 (a42638) | 15.28 / 15.08 (a42638) |
| prose greedy, ms/tok | 23.22 / 23.36 | 23.29 / 23.26 | 24.31 / 24.48 | 24.57 / 24.15 |

- **Exact and free on greedy traffic:** at temperature 0 the drafts are the slice argmax either way, so every greedy
  cell has identical hashes and speed (the extra full-vocab sampling/cache work per draft step does not show).
- **K=5 sampled code: +0.28 accepted per step, −5.5 % ms/tok** (top of the hypothesis range, −2…−5 %). Sampled
  outputs are reproducible across starts (fixed seeds). It recovers 60 % of §5l's greedy→sampled gap at K=5.
- **K=3: +0.03 accepted, null speed** — the gap to recover is small at K=3, as §5l predicted.
- **Best static config for sampled code traffic:** K=5 + probabilistic: 15.9–16.1 ms/tok (~62.5 tok/s), vs 18.0 at
  today's K=3 greedy drafts (−11.5 %). On prose K=3 stays better (§5l). Not a prod change without the user.

### 5o. Stacked: K=5 + probabilistic drafting (exact) −10 % code; + NVFP4 GDN −17 % code (71 tok/s greedy, 68 sampled)

The morning decision table. Three configs on the same PR-equivalent stack (RecoverSSM + align + prefix cache,
`--use-replayssm`), codeprobe (§5l; first-pass numbers incl. cold PLE pages, compare within the table), 2 starts.
Hypothesis `tools/stack/HYPOTHESIS.md`. Data `data/stack/`.

| ms/tok (tok/s) | today: K=3, greedy drafts, FP8 GDN | K=5 + probabilistic, FP8 GDN (exact) | + NVFP4 GDN (quality trade §5j) |
|---|---|---|---|
| code greedy | 16.94 / 16.99 (59.0) | 15.25 / 15.18 (65.7), **−10.2 %** | 14.02 / 14.03 (**71.3**), **−17.3 %** |
| code sampled (1.0/0.95/20) | 17.86 / 17.93 (56.0) | 16.07 / 15.96 (62.4), **−10.5 %** | 14.75 / 14.74 (**67.8**), **−17.6 %** |
| prose greedy | 23.25 / 23.17 (43.1) | 24.43 / 24.18 (41.1), +4.8 % | 23.45 / 23.60 (42.5), +1.4 % |
| code accept_len (greedy / sampled) | 3.36 / 3.15 | 4.35 / 4.17 | 4.42 / 4.26 |
| code c=4 tok/s | 133.9 / 130.9 | 138.3 / 135.0 | 136.7 / 124.5 (c=4 text drifted in start 2) |

- The levers stack roughly multiplicatively, slightly better than predicted on code (−17 % vs −15 %) and slightly
  worse on prose (+1.4 % vs −1 %). NVFP4 GDN does not lower K=5 acceptance on code; it moves it up a little.
- Greedy outputs of the middle column are identical to K=5 greedy drafts (§5n) and, being exact spec decoding, to the
  target model's own output; the right column changes the target (NLL +0.40 % vs BF16, +0.58 % vs FP8).
- Prose still prefers K=3 (§5l). A per-request or adaptive depth would remove that trade-off.
- **Decisions for the user (none taken):** prod K (3 → 5, needs `--block-size 1728`), probabilistic drafting (code
  patch `tools/dprob/`, env-gated), NVFP4 GDN (checkpoint `mtpfp4-gdn4`, quality trade), and updating the prod venv
  from the env-gated RecoverSSM to the PR code (`--use-replayssm`).

### 5p. Adaptive verification at c=1 (vLLM's AdaptiveVerificationManager on MTP + RecoverSSM): rejected — trims, but slower on prose

Question from §5l: can a per-step verify budget give K=5 on code and K≈3 on prose? `tools/avpw/` enables vLLM's
adaptive verification (DSpark-only upstream) under PIECEWISE (`FN_AV_PW`, c=1 only: the QSA builder mis-indexes ragged
verify rows at c>1). Two findings before the run: tail capture sizes above K+1 overflow the RecoverSSM window (v1 crash;
v2 profiles capture sizes only), and the **profiled verify curve is flat for an MoE** (dummy tokens route every row to the
same experts, ~26 ms for 1..6 rows), so v2 never trimmed. v3 injects a measured-shape curve
`FN_AV_VERIFY_CURVE` = 28.1 + 3.3·(n−1) ms. codeprobe_c1, 2 starts, KV 4 GiB. Hypothesis `tools/avpw/HYPOTHESIS.md`.
Data `data/avpw3/`.

| ms/tok (accept_len) | k3 (fixed) | k5 (fixed) | k5av (adaptive) |
|---|---|---|---|
| code greedy | 16.84 / 17.02 (3.36) | **15.13 / 15.19** (4.35) | 15.24 / 15.15 (4.30) |
| code sampled | 17.94 / 18.07 (3.15) | 17.03 / 16.91 (3.89) | 16.85 / 17.18 (3.92 / 3.73) |
| prose greedy | **23.30 / 23.49** (2.53) | 24.18 / 24.23 (2.83) | 25.03 / 24.91 (2.48 / 2.46) |

- **Outside H1 on prose** (expected within +0…+3 % of k3): k5av is +7 % vs k3 and **+3 % vs fixed K=5**. The trim is
  real (prose accept_len 2.83 → 2.47), but all 5 draft steps still run, the cut verify rows cost less than the tokens
  they would have accepted, and the per-step budget adds a host sync. Code: null vs fixed K=5 (within ±1 %).
- Output: k5 hashes reproduce across starts and equal the stack's K=5 run (§5o: code a426386d…, prose f64e554b…);
  k5av code differs from k5 (commit grouping, §5b) and its prose differs between starts (the budget depends on
  confidences that ride the drift).
- **Verdict:** adaptive verification does not pay at c=1 on this stack; K stays a static choice (§5l). What could still
  win is cutting *draft* steps, not verify rows — TensorFold's confidence-stopped chain — but §5b shows a per-step host
  sync costs more than an NVFP4 draft step, so any stop rule must decide on the GPU. Not pursued without an offline
  replay that predicts a gain first.

### 5q. The byte floor at K=5 (estimate): ~53–54 ms per verify cycle; code measures 1.21–1.25×, prose 1.26–1.29×

The 45.2 ms floor (step 2a) is for K=3: a 4-row verify with 26.6 distinct experts per layer. At K=5 the verify has 6 rows
and the drafter runs 5 steps. Scaled from step 2a's measured parts (no new capture):
- **Distinct experts at 6 rows: ~37.2 per layer**, interpolated on the measured sub-linear curve (4 → 26.6, 8 → 47.3;
  exponent 0.83). Not measured: a capture at K=5 would replace this number. Target experts 16.1 → **22.5 ms**.
- **Drafter 3.8 → 5.0–6.5 ms** (5 single-row steps instead of 3 plus a 6-row absorb; the NVFP4 slice head is smaller than
  the head step 2a measured, so the low end is plausible).
- Unchanged: target dense 21.3, lm_head 2.9, GDN state 1.0, QSA 0.2.

| | floor per cycle | measured per cycle (§5o, K=5) | ratio | floor per token |
|---|---|---|---|---|
| code (4.35 accepted) | 52.9–54.4 ms | 15.2 × 4.35 = 66.1 ms | **1.21–1.25×** | 12.2–12.5 ms (80–82 tok/s) |
| prose (2.83 accepted) | 52.9–54.4 ms | 24.2 × 2.83 = 68.5 ms | 1.26–1.29× | 18.7–19.2 ms |

- The ratio is about where K=3 was (1.20×, §4t): deeper drafting added bytes and time in proportion. The per-token floor
  moves with acceptance, so on code the model could reach ~80 tok/s single-stream at the byte floor; we measure 65.7.
- Caveat: the measured cycle includes the cold PLE pages of a first-pass probe (≈7 %, §5l), so the warm ratio is lower,
  roughly 1.15–1.2×.

### 5r. Quality screen for the two precision cuts: NVFP4 GDN and a bf16 SSM state show no loss on GSM8K / HumanEval; NVFP4 GDN costs +6 % TTFT

User rule (2026-09-27): no *noticeable* quality loss; the precision cuts need real benchmarks, and the smaller trade wins
because they may multiply. Four arms on the exact prod config (K=5, probabilistic drafting, RecoverSSM, block 1728 in
every arm), KV 4 GiB, 2 starts each: `fp8gdn` (prod), `nvfp4gdn` (GDN projections NVFP4 W4A16, §5j), `bf16ssm`
(`--mamba-ssm-cache-dtype bfloat16`), `both`. Thinking off, greedy, 16 requests at a time: GSM8K test (1,319) and
HumanEval (164, pass@1, scored offline). TTFT at ~7.5k / ~29k tokens with a unique prefix per request. Tools
`tools/evalq/` (hypothesis amended before any run), data `data/evalq/`.

| arm (start 1 / start 2) | GSM8K % | HumanEval /164 | TTFT 8k s | TTFT 30k s |
|---|---|---|---|---|
| fp8gdn (prod) | 95.91 / 96.44 | 159 / 158 | 2.75 / 2.79 | 10.26 / 10.35 |
| nvfp4gdn | 96.13 / 96.06 | 157 / 158 | **2.93 / 2.94** | **10.93 / 10.95** |
| bf16ssm | 96.21 / 96.21 | 159 / 158 | 2.78 / 2.77 | 10.25 / 10.33 |
| both | 96.44 / 96.21 | 158 / 158 | 2.93 / 2.92 | 10.95 / 10.91 |

- **Noise floor:** the two prod starts differ on 19 GSM8K questions (Δ +0.53 pp, p = 0.17): c=16 greedy is not
  batch-invariant. Every cut-vs-prod pair differs on 16–33 questions with Δ between −0.38 and +0.53 pp (McNemar
  p ≥ 0.25). No cut, and not the combination, is distinguishable from prod; nothing compounds at this resolution.
- **The screen is ceiling-limited** (96 %): it rules out a large loss, not a small one. The bf16 state's risk is error
  accumulated over long sequences, which ≤1k-token prompts cannot show. The decision needs the agentic benchmark
  (SWE-bench slices, `tools/swe/`, smoke run queued).
- **Speed:** NVFP4 GDN costs **+6.2…6.6 % TTFT** at both lengths, reproduced, against a ±1 % floor: Marlin W4A16 runs
  the GDN projections as BF16 math after dequantisation where prod runs FP8×FP8 (prefill is compute-bound; decode is
  bandwidth-bound, hence −7 % there, §5j). The bf16 state costs nothing in TTFT.
- The bf16 arms finished GSM8K in ~2,020 s vs ~2,950 s: at a fixed 4 GiB KV the halved Mamba page holds 105,325 instead
  of 77,608 tokens, so more of the 16 requests fit at once. A capacity effect of the test's small KV, not decode speed;
  prod's default KV (626k tokens) does not have this limit.
- **Verdict so far:** both cuts pass the screen. For agent work (TTFT-bound) NVFP4 GDN trades −7 % decode for +6 % TTFT,
  which is roughly a wash; the bf16 state is free on this screen and must prove itself on long context.

### 5s. Forward to nightly `a9eafde59` (266 commits): bit-identical at c=1, but −4.5 % and non-reproducible at c=4

User: "forward prod to nightly and check if patches apply". The prod overlay (36 files) forwarded onto nightly
`0.30.1rc1.dev219+ga9eafde59` (2 conflicts, resolved from our rebased #58439 branch; `tools/nightly219/`), in a clone
venv with its pins (FlashInfer 0.7.0 — `flashinfer-cubin` 0.7.0 is not published, dropped; cutlass-dsl 4.7.1; humming
0.1.16; torch 2.13.0+cu130 kept). Exact prod config, KV 4 GiB, 2 starts per arm, `nvprobe.py` (codeprobe + TTFT +
cache-hit replay). The first run voided on an armrun limitation (an arm's `FN_VENV` fails the log's venv check);
armrun now takes a per-arm `venv`. Data `data/nightly219/`.

| | prod venv 1ea7c63f4 (s1 / s2) | nightly a9eafde59 (s1 / s2) |
|---|---|---|
| greedy hashes c=1 (code / sampled / prose) | 734976c7 / cb354bf1 / 981f3edb, both starts | **identical**, both starts |
| code c=1 ms/tok (accept) | 15.43 / 15.54 (4.27) | 15.49 / 15.45 (4.27) |
| code sampled c=1 | 16.10 / 16.08 | 16.29 / 16.25 (+1.1 %) |
| prose c=1 | 24.60 / 24.71 | 24.85 / 24.79 (+0.6 %) |
| **code c=4 tok/s (accept)** | **145.7 / 145.7 (4.48), hash 9549b531 both** | **139.7 / 138.7 (4.40 / 4.30), hash differs per start** |
| TTFT 8k / 30k s | 2.77 / 10.30, 2.79 / 10.38 | 2.76 / 10.29, 2.78 / 10.29 |
| cache-hit replay | equal (4.6 → 2.0 s) | equal (5.1 → 2.4 s) |

- **c=1:** the nightly produces prod's exact bits; speed within ±1 %. The derived checkpoints load (the
  `MergedColumnParallelLinear` failure seen on 378504a54 is gone at a9eafde59).
- **c=4: a real regression**, −4.5 % in both starts (gap 6 tok/s against a within-arm spread of 0–1) with lower
  acceptance, and the c=4 text is no longer reproducible across starts (prod's is). Candidates from the forward's risk
  list: #58434 (padded prompt tails counted as spec rows), #58275 (mixed FULL-graph capture), #49371 (Mamba prefill
  state saves). Not bisected. The nightly is not a promotion candidate until c=4 is understood.
- **Prod hash question (open since prodval):** prod's hashes differ from the §5o stack run; prodval ran with the default
  KV (626k tokens) and these arms with 4 GiB, and both give 734976c7 — KV size is not the cause. What remains is the
  capture-size list (prod up to 96, §5o up to 24: short prompts run padded inside a graph).

### 5t. Warm agent turns: `--prefix-match-unit 64` cuts recompute −74 % and turn TTFT −35 %, on prod's own venv (vllm#54458 follow-up)

User: "issue 54458 sounds like worth to fix". Scoping (`tools/i54458/design-memo.md`): vLLM main already saves the GDN state
at the exact end of a prompt when the prefix-match unit (default = the 1728-token block) is smaller than the Mamba block,
so the next turn resumes there instead of at the last block boundary. The memo also claimed our prod base lacked a needed
fix (#58368); the control arm below refutes that. Arms (prod config, K=5, KV 4 GiB, 2 starts each): `nvdef` nightly
a9eafde59 default unit, `nvu64` nightly + `--prefix-match-unit 64`, `m1u64` prod venv 1ea7c63f4 (no #58368) + unit 64.
Probe `tools/i54458/turnreplay.py`: (B) two held-out SWE-bench trajectories replayed as a growing prompt, 46 warm turns
(the agent-loop case); (A) a 20k cached prefix + N fresh tokens inserted *before* the prompt's last line. Hypothesis
`tools/i54458/HYPOTHESIS.md`. Data `data/pmu/`.

| | nvdef (s1 / s2) | nvu64 (s1 / s2) | m1u64, control (s1 / s2) |
|---|---|---|---|
| B: recomputed tokens per turn, median (sum) | 2,025 (98,813) both | **518.5 (31,741)** both | **518.5 (31,741)** both |
| B: turn TTFT median / mean s | 0.874 / 0.945, 0.872 / 0.951 | **0.567 / 0.630, 0.568 / 0.630** | 0.573 / 0.634, 0.585 / 0.639 |
| A: N=16 / 256 / 1024 TTFT s | 1.07 / 1.24 / 1.93 | 1.20 / 1.37 / 2.02 | 1.21 / 1.40 / 2.05 |
| final greedy hash | a70f8e4e both | 50fdcf07 both | 50fdcf07 both |

- **B (append-only turns): recompute −74 %, TTFT −35 % median / −33 % mean**, at the top of H (−65…−90 %, −20…−35 %),
  identical across starts. New tokens per turn median 345; the default resumed ~1,700 tokens further back.
- **A (text inserted before the prompt's end):** no gain (the prompt-end state does not match a prompt that diverges
  earlier) and **+0.1 s** per request: the cost of stopping prefill at the prompt end to save the state.
- **The control has the full gain**, so #58368 is not the mechanism on our stack — out of range per the hypothesis ("a gain
  on m1u64 → #58368 is not the mechanism"). The flag alone does it; the memo's claim was wrong.
- Output differs from the default arm (a70f8e4e vs 50fdcf07): resuming at a different point changes chunking, i.e. the
  reduction order; the flag arms agree with each other and across starts.
- **Prod candidate:** `--prefix-match-unit 64` in `FN_EXTRA` — agent turns −35 %, edit-in-the-middle prompts +0.1 s.
  The user's call.

### 5u. SWE-bench (the deciding benchmark): neither NVFP4 GDN nor a bf16 SSM state loses; both stay inside prod's own run-to-run range

The real benchmark for the two precision cuts (user: "NVFP4 GDN need reals becnhmark", "bf16 SSM state also need
benchmark"). mini-swe-agent 2.4.5 on the x86 box, our fixed slices Verified-30 (Python) + Multilingual-28 (Java/JS),
4 instances at a time, reasoning_effort medium, the model's sampling 1.0/0.95/20, max_tokens 16,000. Server: the exact
prod config (K=5, probabilistic drafting, RecoverSSM with the boundary fix, block 1728), 64k context, default KV; 2 runs
per arm. Hypothesis and verdict rule written first (`tools/swe/HYPOTHESIS-swe.md`). Data `data/swe/` (harness reports,
exit statuses).

| resolved /58 (Python + Java/JS) | run 1 | run 2 | context overflows (run 1 / 2) |
|---|---|---|---|
| prod (FP8 GDN, fp32 SSM state) | 48 (24 + 24) | **52** (28 + 24) | 1 / 1 |
| NVFP4 GDN | 51 (28 + 23) | 50 (27 + 23) | 2 / 1 |
| bf16 SSM state | 50 (26 + 24) | 51 (27 + 24) | 3 / 1 |

- **Run-to-run spread of prod itself: 4 instances** (48 vs 52) at temperature 1.0. Both cuts land inside it on both runs.
- **Per instance, both runs pooled:** NVFP4 GDN better than prod on 7 instances, worse on 5 (sign p = 0.77); bf16 state
  better on 6, worse on 5 (p = 1.00). The pre-registered rule (a cut is a noticeable loss only if both its runs sit below
  both prod runs AND losses clearly outnumber wins) is met by neither.
- **Overflows** (prompt + 16k output > 64k): the same two long lombok instances hit both cuts in run 1; run 2 has one per
  arm. No repeated-step loops in the overflowed trajectories (checked for bf16's django-14034: 98 distinct steps).
- **Against Qwen3.8-27B** on the same Java/JS slice (25/28, two runs, temperature 0.6, larger window): Flash-Next 23–24/28;
  the one instance the 27B alone solved in prod's run 1 (lucene-12212) was a context overflow here, not a wrong answer.
- **Verdict: no noticeable quality loss from either cut** on real agentic tasks, which is what the GSM8K/HumanEval screen
  (§5r) could not show on its own. The choice between them is therefore a speed choice: NVFP4 GDN −7 % decode but +6 %
  TTFT (§5j, §5r); bf16 SSM state no TTFT cost, halves the Mamba state (more KV, smaller align blocks possible). The
  user's call. Not measured: long single-context (> 64k) work, and whether the two cuts compound on this benchmark.

### 5v. Offline replay of logged drafts: two-candidate branching does not pay; a confidence stop would (+3…4 % code, +7 % prose) — but only if drafting actually stops early

User: "is it possible to verify two possible draft token on position n", "maybe if we have two draft token with similar
probability?", "ok". One logging run (clone venv, exact prod config K=5 + probabilistic drafting, greedy c=1, codeprobe's
4 code + 4 prose prompts × 700 tokens): per verify step the drafter's top-2 ids and probabilities, the fed tokens and the
target's token per position (`tools/draftlog/`, 1,647 steps). Replay with a cost model from our measurements: step =
43 ms + K × (1.3 ms draft step + 3.3 ms verify row). Data `data/draftlog/`.

| | code | prose |
|---|---|---|
| tokens per step (logged) | 4.28 | 2.77 |
| per-position acceptance | .89 .76 .64 .55 .45 | .70 .46 .30 .19 .12 |
| chain breaks per step | 55 % | 88 % |
| drafter's 2nd choice = target at the break | 45 % | 37 % |
| branch (1 extra row) at position 1 | −3.6 % | −0.2 % |
| branch at the lowest-margin position | −1.9 % | −0.8 % |
| branch only when margin < 0.1 … 0.5 (variable shapes) | −0.1 … −0.3 % | −0.2 … −0.6 % |
| confidence stop, drafting stops at p₁ < τ, τ = 0.6 / 0.7 | **+3.3 / +4.1 %** | **+6.9 / +7.5 %** |
| same, but all 5 draft steps run and only verify rows are cut | +1.7 / +1.8 % (best +1.8) | +2.1 / +1.5 % (best +3.7 at τ 0.5) |

- **Branching: rejected.** A rescued break gains exactly one token (the chain already emits the target's token there), and
  the second choice is right only 37–45 % of the time, below the ~50 % break-even of a 3.3 ms row. The user's
  similar-probability variant is the least bad (≈ 0), not positive.
- **Confidence stop: worth it only with a real early exit.** Most of its gain is the skipped *draft steps*; cutting only
  verify rows (what adaptive verification did, §5p) leaves +1.8 % code / +3.7 % prose. §5b's version stopped drafting
  with a host sync per draft step and lost 5–6 % to the syncs. So the lever is a draft loop that exits on the GPU (e.g. a
  CUDA-graph conditional node around the draft steps), not another host-side rule. Prose at K=5 + stop would also
  recover the K=3-vs-K=5 prose gap (§5l).
- Caveats: greedy only (sampled drafts differ), c=1, the cost model is ours, not measured per variant; one run.
- **τ sweep (user: "do a p sweep for maximum"; `data/draftlog/tau-sweep.txt`), stop-drafting variant, τ = 0.05…0.95:**
  the curve is flat-topped. Code peaks at **τ 0.75, +4.2 %** (3.8 drafts per step), prose at **τ 0.65, +7.6 %** (2.7
  drafts); the best single τ for both is **0.70: +4.1 % code, +7.5 % prose, +5.8 % mean**, and anything in 0.60–0.75
  is within 0.7 points of it. Above 0.8 it falls off (too few drafts). The optimum barely moves with the cost model
  (τ 0.65–0.75 for draft 1.0–1.6 ms and row 2.5–4.0 ms; the gain scales from +3.4 to +7.9 %). TensorFold's 0.3 is far
  below the optimum for this drafter (+0.6 % code). The rows-only variant peaks lower and earlier (code +1.8 % at 0.7,
  prose +4.0 % at 0.45).
- **A lagged stop does not work** (`data/draftlog/lagged.txt`). The sync-free option from the feasibility memo
  (`tools/earlyexit/feasibility-memo.md`): choose the draft count for round n+1 from round n's confidences, which the
  host already has without a sync. Replayed on the same log, observing only drafted positions: code **−0.8 … −8.2 %**
  (worse at every τ), prose **+1.9 % at best** (τ 0.4); a running-product rule is worse still. One round's drafter
  confidence does not predict the next round's, so the whole gain needs the stop *inside* the round, i.e. without a
  host decision: CUDA-graph IF nodes around FULL-captured draft steps (torch 2.13 has `begin_capture_to_if_node`; no
  engine we found does this), or first an nsys check of whether the eager draft steps are launch- or GPU-bound.

### 5w. Weight loading: vllm#58868 cuts the main load −87 % (505–534 s → 65–66 s), model loading 575–599 s → 111–121 s

User: "move loading speed on top since it speeds up everything". [vllm#58868](https://github.com/vllm-project/vllm/pull/58868)
(Willian-Zhang, open) touches one byte per page of an mmap'd checkpoint tensor on the CPU before the host-to-device copy
on integrated GPUs; on GB10 the page faults otherwise happen inside the driver copy, several times slower. Its diff
applies with offsets to our overlaid 1ea7 files (`tools/p58868/`). Clone venv, prod config, **every start from a cold
page cache** (the launcher drops caches), 2 starts per arm, alternating. Hypothesis `tools/p58868/HYPOTHESIS.md`.
Data `data/ld58868/`.

| cold start | main weights | MTP drafter weights | model loading total | greedy sanity hash |
|---|---|---|---|---|
| stock, s1 / s2 | 504.9 / 534.0 s | 62.6 / 54.4 s | 575 / 599 s | 350b6b16 |
| #58868, s1 / s2 | **65.9 / 64.9 s** | 41.4 / 32.5 s | **121 / 111 s** | 350b6b16 |

- **Main load −87 %**, beyond the hypothesis (−30…−55 %), identical output: the same bytes are loaded.
- The drafter still takes 32–41 s: it walks all ~300k checkpoint tensors to keep ~3k. blazux's patch 18 (MTP name
  prefilter, ported to 1ea7 in `tools/fastload/`) targets exactly that; the fastload A/B (queued) measures #58868 +
  their patches 16 and 18 together.
- A prod start drops from ~12 min to ~4 min with this alone. Prod candidate (the user's call); it changes loading only.

**Addendum (2026-09-28): the fast-loading set, in prod since the same day.** #58868 + blazux patch 16 (FusedMoE expert
name index) + patch 18 (MTP name prefilter, ported to 1ea7), `tools/fastload/`. Same method (clone venv, prod config,
cold page cache, 2 starts per arm, alternating). Hypothesis `tools/fastload/HYPOTHESIS.md`. Data `data/fastload/`.

| cold start | main weights | MTP drafter weights | model loading total | greedy sanity hash |
|---|---|---|---|---|
| stock, s1 / s2 | 579.8 / 537.1 s | 61.4 / 57.9 s | 651 / 605 s | 350b6b16 |
| fastload, s1 / s2 | **41.6 / 41.9 s** | **3.75 / 3.48 s** | **56.7 / 56.0 s** | 350b6b16 |

- **Model loading −91 %** (11 → 1 min), output identical. On top of #58868 alone (65–66 / 32–41 s above) the expert
  name index takes the main load −36 % and the prefilter takes the drafter to 3.5–3.75 s.
- Against the hypothesis: the main load (−92 %) is beyond its −40…−70 % range, because #58868 alone already exceeded
  its own. The prefilter kept **6,176** tensors, not the ~3.1k predicted; the prediction's count was wrong, not the
  filter (drafter load −94 %, same hash). Not checked which tensors make up the difference.
- blazux's own full set (with patches 14/15/17, v0.30) reports main 35.5 s; the remaining ~6 s is what 15 (pread for
  small tensors) and 17 (chunked embedding copy) could still take. Low priority now: loading is ~1 min of a start.
- Installed in prod's venv 2026-09-28 (user: "yes, promote and post"); GB10 numbers for #58868 posted on the PR.

### 5x. vllm#53912 on our stack: a decode-written recurrent state is never served from the prefix cache (probe void, mechanism found)

The contamination check (decode-written Mamba blocks read back through the prefix cache; `tools/decblk/`, hypothesis
there) came back **void by its own rule twice**: v1 on token-id prompts, then v2 (chat endpoint, `continue_final_message`)
in both arms. The follow-up never hit past the first request's prompt (0 of 16 seeds, spec and nospec), while the
probe's own diagnostic shows the cache working (an identical prompt's first repeat hits 0, its second hits 3,456
tokens). Data `data/decblk2/`.

**Correction (same day, decblk3's start failed on it):** the scheduler block on this model is **3,456 tokens** (2 × 1,728),
and cache hits come in that unit (the diagnostic's hit was exactly 3,456). The v1/v2 follow-ups (1,871 and 3,127 tokens)
were too short to hit at all, so **the void was the probe's length**; the retention rule below is from reading the code
and was not what this probe measured. decblk3b (380-number prompt, follow-up ~3.57k crossing 3,456 during decode,
retention interval 3,456) is queued as `night31`.
**decblk3b (same day): void again.** Prompt 3,071, follow-up 3,551; B hit 0 (spec) and 1,728 (nospec, a prompt-only
boundary), never 3,456, R missed in all. The 3,456 boundary lies inside the generated tokens, and v2/v3 rebuild the
follow-up by re-tokenizing the generated text, so a token mismatch there is not excluded. decblk4 (v1's exact token-id
method, same long prompt and interval) is queued as `night33`. Data `data/decblk3b/`.

**decblk4 (same day): valid, and no contamination.** v1's exact token-id method, prompt 3,047 tokens, follow-up 3,527,
`--prefix-cache-retention-interval 3456`, 16 seeds per arm. B hit **3,456 tokens in 16/16** in both arms, past the prompt:
a recurrent state written during (speculative) decode was served. R missed in all. So the v2/v3 voids were the
re-tokenized follow-up.

| arm | B hits | divergent / 16 | first divergence (token) |
|---|---|---|---|
| spec (MTP K=5 + RecoverSSM + nodrop) | 3,456 × 16 | 4 | 19, 6, 8, 22 |
| nospec (native align) | 3,456 × 16 | 2 | 7, 7 |

- **H0 holds** (spec ≤ nospec + 2: 4 ≤ 4); H1 (≥ +4 and divergence at tokens 0–2) is not met. The divergence is late
  and present without speculation too: cached-vs-recomputed numeric drift, not #53912's poisoned state.
- Prod runs retention interval 0, which this did not test; decblk5 (interval 0, spec arm) is queued as `night35`.
  Data `data/decblk4/`.

**decblk5 (prod's interval 0, spec arm, same probe): B hit 0 tokens in 16/16**, R 0, 0 divergent. As the code predicted,
the decode-crossed boundary is not retained at the default, so **in prod a decode-written recurrent state is never
served: #53912's path is unreachable (measured)**; where a config does retain it (decblk4), it did not contaminate.
Data `data/decblk5/`.

From the code (`v1/core/single_type_kv_cache_manager.py`, reachable-boundary mask; `config/cache.py`): align-mode
Mamba retains recurrent-state snapshots **sparsely**. `--prefix-cache-retention-interval` defaults to **0 = "only
semantic checkpoints"**: the latest replay boundary (prompt end) and shared-prefix junctions. A block boundary crossed
during decode is not retained, so there is nothing to hit, with or without speculation. The same rule explains the old
"hits only from the second repetition" observation: the first repeat creates the shared-prefix junction, the second reads it.

- **For prod (interval unset): a decode-written state is not retained** (code, then measured by decblk5 below), so
  #53912's path is not reached. Where it is reached (interval 3,456, decblk4), it did not contaminate.
- det-236 (09-24, "15 reads into decode-written blocks, 0 divergent") ran on the older stack; whether its reads really
  hit decode-written states was not re-checked against this mechanism, so it no longer counts as evidence either way.

**decblk6 (2026-09-29, #53912 at prod's `--prefix-match-unit 64`): the fine-grained path IS reached, and speculation
adds no divergence.** decblk5 above ran without `--prefix-match-unit 64`, which prod has used since 09-28; with it, the
Mamba lookup takes the fine-grained branch cch-zuzuche flagged on #53912 (it returns the prompt's partial-tail entry
and ignores `drop_eagle_block`; the same code is in our venv). Same probe (decblk4.py, 16 seeds), prod flags; spec =
MTP K5 + RecoverSSM + nodrop, nospec = the same flags without speculation; 1 start each.

| arm | B hits | divergent / 16 | first divergence (token) |
|---|---|---|---|
| spec | 3,008 × 16 | 5 | 22, 9, 2, 11, 3 |
| nospec | 3,008 × 16 | 5 | 8, 13, 1, 8, 0 |

- **B hits 3,008 = the prompt's last 64-token boundary in 16/16**, so decblk5's "unreachable" no longer holds for prod
  flags: the partial-tail state is served. It is a prefill-written state (no decode, no drafts in it).
- **H0 holds on the count** (spec 5 ≤ nospec 5 + 2). H1's "divergence at tokens 0–2" clause fired, but in **both**
  arms, and earlier without speculation (token 0 and 1): it is cache-hit vs recompute numeric drift, not a
  speculation effect. Three seeds (2, 6, 13) diverge in both arms. The continuation is random 7-digit numbers, i.e.
  near-ties at every digit, which is why a small drift shows up at once.
- Not covered: the concurrent producer/consumer copy-on-write case (a sibling hitting a partial tail while its producer
  still decodes in that block) and cch-zuzuche's shape (0 % acceptance episodes on 85k+ reused agent prefixes). No
  degeneration was seen here, but this probe compares tokens, not text quality. Data `data/decblk6/`.

### 5y. Validation of the 2026-09-28 prod config (bf16 SSM state, `--prefix-match-unit 64`, fast loading, tool guards)

One start with prod's venv, launcher and drop-in flags (armrun `prodval2`, default KV size, as the 09-27 `prodval`),
13 required path lines present (incl. `'prefix_match_unit': 64`, `mamba_ssm_cache_dtype': 'bfloat16'`, the MTP name
prefilter), 5 forbidden absent. Data `data/prodval2/`.

| | 09-27 config (`prodval`) | 09-28 config (`prodval2`) |
|---|---|---|
| code, c=1 greedy | 15.37 ms/tok | 14.74 ms/tok (accept 4.37) |
| code, c=1 sampled | 16.14 ms/tok | 16.60 ms/tok |
| prose, c=1 greedy | 24.72 ms/tok | 23.66 ms/tok |
| code, c=4 | 134.9 tok/s | 144.8 tok/s |
| TTFT 8k / 30k (cold, nonce) | — | 2.89 / 10.39 s |
| cache-hit replay (same 8k twice) | — | 5.02 → 2.10 s, output equal |
| start to ready | ~12 min | 2 min 39 s |

- **Validates the config**: every path taken, cache replay equal, nothing slower beyond single-start spread.
- **Not an A/B**: one start per config, and restarts alone move c=1 by up to 11 % (§4c, cold window), so the
  −4 % / +7 % differences are not attributed to any of the four changes. Greedy hashes differ from 09-27, as
  expected with a bf16 state (a precision change; SWE-bench showed no loss, §5u).

### 5z. F4: full CUDA graphs for the RecoverSSM verify path — −1.1…−2.1 % short-context decode, no regression (the 8k "regression" was a probe artefact, withdrawn below)

User: "do 4. next", then "all". The GDN and PLE RecoverSSM builders now declare `UNIFORM_BATCH`; graph padding rows
get null state slots and zero-length windows (the verify kernel writes zeros and returns for them). Kernel and config
tests: 91 passed, including a new padded-row test (`data/f4/f4-pytest2.log`). Clone venv `rssm`, prod config (K=5,
probabilistic drafts, bf16 state, prefix-match-unit 64), KV 4 GiB, `FN_CG_MODE` FULL_AND_PIECEWISE vs PIECEWISE,
2 starts each, alternating; void rules on `Capturing CUDA graphs (FULL)` / its absence. Hypothesis `tools/f4/HYPOTHESIS.md`.
Data `data/f4/`.

| | full, s1 / s2 | piecewise, s1 / s2 | Δ |
|---|---|---|---|
| code c=1 greedy, ms/tok | 14.723 / 14.695 | 15.003 / 14.882 | **−1.1…−2.1 %** (sign holds both rounds) |
| code c=1 sampled, ms/tok | 16.596 / 16.502 | 16.731 / 16.687 | −0.8…−1.4 % |
| prose c=1 greedy, ms/tok | 23.767 / 23.570 | 23.673 / 23.727 | null |
| code c=4, tok/s | 143.56 / 142.86 | 142.70 / 143.51 | null |
| TTFT 8k / 30k, s | 2.875, 10.372 / 2.866, 10.355 | 2.922, 10.510 / 2.962, 10.419 | −1…−3 % (prefill is piecewise in both) |
| 8k prompt + 96 tokens, cold / cache-hit, s | 4.908, 1.968 / 5.147, 2.067 | 4.564, 1.599 / 4.563, 1.744 | **+7…+13 % / +13…+29 %** |

- **Correct:** greedy hashes identical between arms at c=1 (code and prose); the c=4 code hash differs in one full start,
  within the known non-batch-invariance under concurrency.
- **Against the hypothesis:** short-context c=1 inside the predicted −0…−4 %; c=4 null as predicted. The 8k-context
  request is out of range: its cache-hit repeat is mostly decode (96 tokens), and it is 0.22–0.47 s slower with full
  graphs in both starts (gap beats each arm's spread). Same direction as finding 237 (FULL_DECODE_ONLY: agent turns
  3–4 % slower). Mechanism not yet known; the lead is a context-dependent decode cost under full graphs (graph-captured
  attention/QSA launched for the capture shape rather than the live context).
- ~~**Verdict: not a prod candidate** as is.~~ Superseded by the withdrawal below. Agent work runs at long context. Next: a decode-vs-context sweep, both arms
  (`tools/f4/`, queued), to locate the regression before any upstream follow-up.

**§5z addendum — decode vs context (`f4ctx`, same arms, 2 starts each): no context-dependent decode cost.** Streamed
decode (prefill excluded), greedy, 256 tokens, 2 fixed prompts per size; hashes identical between arms at every size.

| context | full, ms/tok (req 1, req 2; s1 / s2) | piecewise | Δ |
|---|---|---|---|
| 1k | 22.01, 19.37 / 21.91, 18.47 | 21.99, 18.65 / 21.93, 18.64 | null |
| 8k | 19.49, 18.42 / 19.24, 18.26 | 19.68, 18.56 / 19.63, 18.37 | ≈ −1 % |
| 16k | 18.93, 18.53 / 19.35, 18.58 | 19.12, 18.91 / 19.27, 18.72 | ≈ null |
| 28k | 19.59, 15.81 / 19.28, 15.87 | 19.57, 16.15 / 19.59, 16.35 | 0…−3 % |

TTFT equal at every size (e.g. 28k: 9.80–9.85 vs 9.82–9.87 s). **H-alt holds**: the §5z 8k-request slowdown is not
decode at long context. Still unexplained: that request (8k prompt + 96 tokens, run after the TTFT probes) paid a fixed
+0.35…0.5 s in both full starts. Its output was not hashed or token-counted across arms, so a different output length is
not excluded. Next: a replay probe that streams, counts and hashes, run both before and after the TTFT probes.
Data `data/f4/f4ctx.*`.

**§5z withdrawal — the 8k-request regression was the probe (`f4rep`, same arms, 2 starts).** nvprobe's replay prompt
carries a random `uuid4` nonce, so every arm and start answered a *different* prompt, with a different text and length.
With fixed prompts (3 replay pairs on the fresh server, the TTFT probes, 3 more pairs), outputs, token counts (76–96)
and hashes are identical between arms, and the output length alone moves the cache-hit request between 1.69 and
2.30 s: the whole §5z spread. Full vs piecewise on the same prompts:

| | full, s1 / s2 | piecewise, s1 / s2 | Δ |
|---|---|---|---|
| cache-hit replay, phase A (3 prompts), s | 2.175, 2.257, 1.688 / 2.186, 2.265, 1.692 | 2.223, 2.304, 1.721 / 2.221, 2.303, 1.721 | −1.5…−2.2 % |
| cache-hit replay, phase B (after the TTFT probes), s | 1.808, 2.093, 2.255 / 1.818, 2.100, 2.269 | 1.846, 2.136, 2.304 / 1.846, 2.137, 2.301 | −1.4…−2.1 % |
| TTFT 8k (median of 3) / 30k, s | 2.773, 10.180 / 2.789, 10.214 | 2.845, 10.429 / 2.808, 10.312 | −0.7…−2.5 % / −1…−2.4 % |

- **H1 (output length) confirmed; H2 (state after long prefills) and H3 (per-request cost) refuted.** The §5z row
  "8k prompt + 96 tokens … +7…+13 % / +13…+29 %" is **withdrawn**. nvprobe's replay now uses a fixed prompt, so its
  replay times compare across arms (earlier nvprobe replay numbers compare only within one server).
- **Revised F4 verdict:** short-context c=1 −1.1…−2.1 %, replay and TTFT −1…−2.5 %, everything else null, outputs
  identical at c=1. **A prod candidate (the user's call)** and an upstream follow-up to #58863. It also un-parks the
  GPU-side early exit, which needs the draft steps under full graphs. Data `data/f4/f4rep.*`.

### 5aa. Byte ledger (TODO 5b): at prefill, ~15–20 % of TTFT is intermediates that round-trip DRAM; hyper-connections are the largest

> **Prior art (found 2026-09-28, after these measurements):** TensorFold shipped the same HC fusion two days earlier
> (`_qmm_hcdown` / `_qmm_upmix`, 0.3.0, 2026-09-26) for 4-bit weights; it stores `xn` once where ours rebuilds it.
> Details and the provenance check in [the-field](the-field.md). Our work was independent; cite theirs upstream.

User: "do we have fusable kernels where a fusion would lower bytes read/written? I think we checked only for launch
overhead". §4m and §5f priced fusions by launches and by avoidable work; this prices them by **bytes**. One nsys trace
on the prod config (clone venv, PIECEWISE, K=5, bf16 state, KV 4 GiB): a 7,507-token prefill (chunks 3,456 + 3,456 +
tail) and a 64-token decode window. GB10's ncu has no DRAM byte counters, so bytes come from the launch grids and the
model's shapes (hidden 2,560 per stream × 4 hyper-connection streams = 10,240; MoE top-10 of 512, intermediate 640).
Hypothesis `tools/ledger/HYPOTHESIS.md`; script `tools/ledger/ledger_prefill.py`; data `data/ledger/`.

**The memory-bound families run at the DRAM limit**, so their time *is* their bytes (per 3,456-token chunk):

| kernel (calls in the 2,676 ms prefill window) | bytes per call | median per call | rate | share of prefill |
|---|---|---|---|---|
| `_hc_combine_norm` (330): reads residual + block out, writes new residual **and** normalized `xn` | 230 MB | 0.98 ms | 234 GB/s | 7.9 % |
| `_hc_gate_mix` (336): reads `xn` + `gate`, writes block input | 159 MB | 0.73 ms | 219 GB/s | 5.8 % |
| `finalizeMoeRouting` (159): reads 10 expert rows per token, writes one | 195 MB | 0.87 ms | 224 GB/s | 3.5 % |
| `doActivation` (159): GEMM1 out → SwiGLU | 132 MB | 0.49 ms | 270 GB/s | 2.0 % |
| `expandInputRows`, `cvt_fp16_to_fp4`, prefix sums (MoE glue) | | | | 2.3 % |
| `layer_norm_fwd`, `per_token_group_quant`, elementwise glue | | | | ~6 % |

**Where fusion removes bytes (prefill, ranked):**

1. **Hyper-connections, ~7 % of TTFT (≈ 0.2 s of 2.7 s at 7.5k).** Each HC block per chunk moves 531 MB: `combine_norm`
   writes a normalized copy `xn` (71 MB) that the down-projection and `gate_mix` each read back, and the up-projection
   writes `gate` (71 MB) that `gate_mix` reads back. Fused design, same math (bit-identical for the combine/norm part; the up GEMM moves from cuBLAS to our own kernel, so
   `gate` changes at accumulation-order level, like any kernel swap: drift-sized, checked against an fp32 reference):
   `combine_norm` writes only the residual plus one rrms per (row, stream); the down GEMM normalizes on its A-load;
   `gate_mix` moves into the up GEMM's epilogue (reorder W_up's output columns offline so the 4 streams of one index
   sit in one tile, round the gate to bf16 before the sigmoid as today). 531 → 318 MB per block: **−0.93 ms per block
   per 3.5k chunk, × 196 blocks + the tail ≈ −0.2 s per 7.5k prefill.** Kernel work: one Triton GEMM with a custom
   epilogue (K = 320, small) and a trimmed `combine_norm`.
2. **MoE GEMM1 → SwiGLU → fp4 in one epilogue, ~4 %** (finding 144's "1.1 ms per layer"): removes GEMM1's output round
   trip and the bf16 activation (≈ 264 MB per layer-chunk). FlashInfer CUTLASS kernel work, the hardest item.
3. **Fused finalize, ~3–5 %, but it is a determinism trade we chose.** Prod's `VLLM_MOE_DET_FINALIZE=1` (bit-stable
   finalize overlay) sets `use_fused_finalize=False`, so GEMM2 writes all 10 expert rows per token (177 MB) and a
   separate pass reads them back. The fused (atomic) finalize removes that, at the cost of run-to-run bit stability
   (finding 145). A deterministic fused finalize (each token's 10 rows reduced in a fixed order inside the GEMM2
   epilogue) would get both; not available today.
4. **Expand-rows gather into GEMM1's A-load, ~1.5 %**; norm/quant into producers' epilogues, ~1–2 %.

**Decode (6-row verify): no byte prize.** The same HC block moves 6 × 88 KB ≈ 0.5 MB, L2-resident; intermediates are
< 1 % of a verify cycle's bytes (weights dominate), as §5f found. Byte fusion is a **TTFT** lever, and agent speed is
TTFT-bound.

- **Against the hypothesis:** total prefill prize in range (predicted 8–15 %; ranked items 1–4 sum to ~15–20 % if all
  were built, ~11 % without the determinism trade). HC is larger than predicted (7 % vs 3–6 %), the MoE chain smaller
  (4 % vs 5–9 %). Decode < 1 % as predicted.
- **Next (proposed):** item 1, standalone first. A Triton up-GEMM with the gate-mix epilogue plus the trimmed
  `combine_norm`: `combine_norm` outputs bit-exact, block input within today's own rounding error of an fp32 reference, at M = 3,456 and M = 6, timed standalone; a server A/B
  on TTFT only if the standalone saves ≥ 0.6 ms per block.

**§5aa addendum — HC fusion standalone (`tools/hcfuse/hcfuse.py`, hypothesis there): −0.92 ms per block at prefill, drift-level numerics.**
One hyper-connection block with injection, random bf16 data at the model's shapes, median of 30, 4 tile configs:

| | today (combine_norm → cuBLAS down → silu → cuBLAS up → gate_mix) | fused (K1 residual + rrms → K2 down, normalize on load → silu → K3 up + gate-mix epilogue) |
|---|---|---|
| M = 3,456 (prefill chunk) | 2.706 ms | **1.788 ms** (best of 4 configs; others 1.84–2.06) = **−0.92 ms, −34 %** |
| M = 6 (decode) | 0.062 ms | 0.113–0.302 ms (slower) |
| residual, `xn` | | bit-exact at both sizes |
| block input vs today | | **identical at M = 3,456** (max diff 0.0); 1 bf16 ulp at M = 6 |
| block input vs fp32 reference (max / mean) | 0.0101 / 5.46e-4 | 0.0101 / 5.46e-4 (same) |

- In the hypothesis range (−0.6…−1.2 ms per block). Numerics: identical to today at M = 3,456 in this test, but the
  registered op's test (`tools/hcfuse/test_hcfuse_op.py`, `data/hcfuse/`) shows 1 bf16 ulp in the down GEMM's output and
  the block input at M = 512 and 595 (Triton vs cuBLAS accumulation order), so **drift-level, not bit-identical**. Below
  the size threshold the op runs today's kernels and is bit-identical (M = 6, 96, 511).
- **Dispatch by size:** fused only for prefill chunks (slower at decode). The size branch must live inside a custom op,
  not in traced Python (memory `vllm-compile-freezes-branches`).
- Projected at the server: −0.92 ms × ~196 blocks + the tail ≈ **−0.19 s per 7.5k prefill (≈ −6.7 % TTFT)**. Next:
  an env-gated overlay on the clone venv and a TTFT A/B with decode hashes as the control.

**§5aa server A/B — HC fusion (`FN_HCFUSE=1`) on the prod config: TTFT −5.5…−7.9 %, decode unchanged, outputs identical.**
Clone venv with the overlay (`tools/hcfuse/`: custom op, fused path at ≥ 512 tokens per batch, today's kernels below),
PIECEWISE, K=5, probabilistic drafts, bf16 state, pmu 64, KV 4 GiB; 2 starts per arm, alternating; void rules on the
op's log line (present / absent). Probe nvprobe (fixed replay prompt). Data `data/hcfuse/hcfuse-ab.*`.

| | hcfuse, s1 / s2 | base, s1 / s2 | Δ (sign holds both rounds) |
|---|---|---|---|
| TTFT 8k, median of 3 | 2.735 / 2.760 s | 2.921 / 2.970 s | **−5.5…−7.9 %** |
| TTFT 30k, median of 3 | 9.773 / 9.817 s | 10.465 / 10.595 s | **−6.2…−7.8 %** |
| 8k replay, first send / cache hit | 4.245, 1.596 / 4.276, 1.598 s | 4.550, 1.655 / 4.477, 1.652 s | −4.5…−6.7 % / −3.3…−3.6 % |
| code c=1 greedy / sampled, ms/tok | 14.825, 16.614 / 14.794, 16.733 | 14.863, 16.923 / 14.861, 16.764 | null |
| prose c=1, ms/tok | 23.477 / 23.871 | 24.205 / 24.300 | −1.4…−3.4 % |
| code c=4, tok/s | 145.4 / 141.96 | 144.4 / 143.58 | null |
| greedy hashes (code, prose) | d102a738, 38c70791 | same | identical |

- **In the hypothesis range** for both TTFT cells (−4…−8 % / −4…−9 %) and decode; the saving scales with prompt length,
  as the byte model predicts (~0.19 s at 7.5k, ~0.7 s at 29k).
- **Slightly beyond it, favourable:** the cache-hit replay (its uncached suffix is ~595 tokens, above the 512 threshold,
  so it takes the fused path) and prose c=1 (−1.4…−3.4 %, sign holds; the probe's time includes a prompt prefill).
  Not attributed further.
- **Correctness:** no output change on the probes; the op is drift-level by its own test (1 bf16 ulp at some sizes).
- **Status:** clone venv only. **Prod candidate (the user's call)** — the largest TTFT lever measured this week — and an
  upstream candidate (the HC ops live in vLLM's `qwen4_exp`). Items 2–3 of the ranking (MoE epilogue, deterministic
  fused finalize) stay open.

**§5aa — HC fusion size threshold (`tools/hcfuse/sweep_hcfuse.py`): crossover at 640 tokens, not 200–400.** The op
forced fused vs today's sequence, one block, median of 50:

| M | 16–64 | 96 | 128 | 256 | 384 | 512 | 576 | 608 | 640 | 704 | 1024 | 1536 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused vs today | +203…+227 % | +143 % | +102 % | +24 % | +11 % | 0…+6 % | +1.4 % | +0.6 % | **−11.5 %** | −16.6 % | −29.2 % | −27.6 % |

- **Out of the hypothesis range** (predicted crossover 200–400). The fused path has a ~0.195 ms floor: the down GEMM
  launches only cdiv(M, 64) × 3 programs, each looping over K = 10,240, so below ~600 tokens it is latency-bound. The
  jump at 640 is on today's side (cuBLAS changes tile: 0.40 → 0.47 ms).
- **Default threshold moved 512 → 640** (clone venv and `tools/hcfuse/hc_fused.py`). It changes only batches of
  512–639 tokens (≤ 1.4 % of one block), so the A/B above stands.
- **Follow-up:** split-K in the down GEMM would lower the floor and extend the win to agent-turn suffixes (a few hundred
  tokens with pmu 64), where most prefill in the agent loop happens.

**§5aa — HC fusion split-K (`tools/hcfuse/splitk_hcfuse.py`): the fused path now wins from 128 tokens.** The down GEMM's K
(10,240) split into 4 or 8 slices, fp32 partials reduced in a fixed order (deterministic, no atomics):

| M | 64 | 128 | 192 | 256 | 384 | 512 | 640 | 1024 | 3,456 |
|---|---|---|---|---|---|---|---|---|---|
| today, ms | 0.066 | 0.105 | 0.156 | 0.203 | 0.284 | 0.350 | 0.421 | 0.724 | 2.663 |
| best fused, ms (split) | 0.071 (8) | 0.082 (8) | 0.130 (4) | 0.159 (4) | 0.230 (4) | 0.297 (4) | 0.375 (8) | 0.548 (4) | 1.755 (1) |
| Δ | +8 % | **−21 %** | −17 % | −22 % | −19 % | −15 % | −11 % | −24 % | −34 % |

Every split reproduces bit-for-bit run to run; output within 1 bf16 ulp of today's. The op now runs today's kernels
below 128 tokens and splits 8 / 4 / 1 below 192 / 2,048 / above; its test (`data/hcfuse/hcfuse-optest2.txt`) passes at
6…3,456 (bit-identical below 128, identical at 2,048 and 3,456, 1 ulp between). Better than the hypothesis (crossover
128–256). An agent-turn A/B (§5t's turn replay) is queued; the 8k/30k TTFT cells above are unaffected (chunks ≥ 2,048
run unsplit, as measured).

**§5aa — HC fusion on agent turns (`hcturn`, §5t's turn replay, split-K op, 2 starts per arm): small.**

| | hcfuse, s1 / s2 | base, s1 / s2 | Δ |
|---|---|---|---|
| A: 20k cached + 16 new tokens, TTFT | 0.655 / 0.670 s | 0.680 / 0.680 s | −1.5…−3.7 % |
| A: + 256 new tokens | 0.839 / 0.851 s | 0.884 / 0.879 s | −3.2…−5.1 % |
| A: + 1,024 new tokens | 1.448 / 1.464 s | 1.530 / 1.520 s | −3.7…−5.4 % |
| B: 46 replayed SWE-bench turns, TTFT median / mean | 0.566, 0.619 / 0.568, 0.618 s | 0.572, 0.630 / 0.563, 0.623 s | **null** / −0.8…−1.9 % |
| recomputed tokens (A cells; B median / sum) | 1,120 / 1,639 / 3,300; 482.5 / 28,989 | same | identical |
| B final greedy hash | a70f8e4e (both starts) | cfb9b78f (both starts) | differs, stable within arm |

- **Against the hypothesis:** A in range (−2…−6 %); **B below it** (predicted −2…−5 %). A median turn recomputes ~480
  tokens: the fused path saves ~0.05 ms per HC block there (−15 % of ~0.33 ms), × 98 blocks ≈ 5 ms of a 0.57 s turn.
  The turn is dominated by costs the HC blocks don't touch. The HC fusion is a **long-prefill** lever (cold 8k/30k
  −6…−8 %, above), worth ~1 % on warm agent turns.
- **Output:** the replay's final greedy text differs between arms and is reproducible within each: the op's 1-ulp
  numerics flip a greedy token somewhere in 46 turns. Deterministic drift, the class the quality rule accepts; a quality
  check (SWE-bench or the logprob screen) would come before any prod use.

### 5ab. Pinning vLLM to the X925 cores (bilikaz's recipe, agenda item 8): no gain; single-stream decode +1.2…1.5 % slower

bilikaz report +2–3 % from `taskset` to the Cortex-X925 cores. On this box those are CPUs 5–9 and 15–19 (MIDR d85,
3.9 GHz; the A725s are 2.8 GHz). Prod's venv and flags, KV 4 GiB, 2 starts per arm, alternating; the pinned arm's
shell logs its own `Cpus_allowed_list` (void rule). Hypothesis `tools/cpuset/HYPOTHESIS.md`; data `data/cpuset/`.

| | pinned, s1 / s2 | free, s1 / s2 | Δ |
|---|---|---|---|
| code c=1 greedy, ms/tok | 14.990 / 15.017 | 14.815 / 14.799 | **+1.2…+1.5 %** (sign holds both rounds) |
| code c=1 sampled | 16.700 / 16.838 | 16.653 / 16.676 | +0.1…+1.1 % |
| prose c=1 | 24.176 / 24.270 | 23.767 / 24.155 | +0.1…+2.1 % |
| code c=4, tok/s | 142.46 / 145.0 | 143.79 / 144.08 | null |
| TTFT 8k / 30k, s | 2.877, 10.342 / 2.887, 10.394 | 2.903, 10.449 / 2.903, 10.475 | −0.6…−0.9 % / −0.8…−1.3 % |
| greedy hashes | identical | | |

- **Out of the hypothesis range** on the slow side (c=1 predicted −0…−3 %). The whole process tree (API server, engine
  core, worker, their thread pools) shares 10 cores when pinned; the cause of the small c=1 loss is not isolated.
  TTFT gains ~1 %, inside noise-adjacent territory.
- **Not adopted.** Their gain may depend on their container's thread settings; on bare metal with default threads it
  does not transfer. Pinning only the worker (not the API server) is untested.

**§5aa — MoE activation fusion via b12x's gated kernel: closed (`tools/b12xg/`, TODO item 6 step 1).** The plan was to
raise b12x's intermediate-size guard (4 × 128 = 512; ours 640) and use its fused FC1 → SwiGLU → FC2-input kernel.
Standalone, random NVFP4 weights at the model's MoE shapes, one process per mode:

- **Run 1 was void:** flashinfer's dynamic-kernel cache key omits the gated decision, so the "gated" wrapper reused the
  generic kernel compiled before the guard was raised (equal times).
- **Runs 2–4 with a witness** (the constructed kernel class, then the selector's own arguments): the gated kernel is
  **never constructed** even with the guard raised. The dispatch picks MMA tiles **(32, 128) at 1,024 tokens and
  (64, 128) at 3,456** from rows per expert (3,456 × 10 / 512 ≈ 67); the gated kernel requires (128, 128). With 512
  experts and ≤ 4,096-token chunks no expert reaches 128 rows, so the intermediate guard was never the binding limit.
- **b12x is not run-to-run deterministic here:** the same call twice in one process differs by 0.65–0.69 % relative
  L2 (generic kernel), which our bit-stable-finalize requirement rules out regardless.
- **Against the hypothesis:** H1 (does not work) holds, for a reason the hypothesis did not name (tile selection, not
  shared memory). The remaining route to the ~4 % is the CUTLASS SM120 EVT epilogue (TODO item 6, step 2).

**§5aa — HC fusion quality screen (`evalhc`, §5r's GSM8K + HumanEval, 2 starts per arm): no measurable change.**
Clone venv, prod config, thinking off, greedy, c = 16; HumanEval scored offline as the unprivileged user
(`data/hcfuse/evalhc-score.txt`).

| | GSM8K (1,319) | HumanEval (164) | TTFT 8k / 30k |
|---|---|---|---|
| hcfuse, s1 / s2 | 96.29 / 96.29 % | 158 / 156 | 2.716, 9.732 / 2.741, 9.808 s |
| base, s1 / s2 | 95.98 / 96.66 % | 157 / 159 | 2.905, 10.416 / 2.928, 10.555 s |

- Every hcfuse-vs-base McNemar p ≥ 0.25 (GSM8K 0.27–0.52, HumanEval 0.25–1.0). The base arm's two starts differ more
  from each other on GSM8K (p = 0.049; c = 16 is not batch-invariant across starts) than either hcfuse start does from base.
- Against the hypothesis: GSM8K inside base's spread; HumanEval start 2 (156) is one item below base's 157–159, start 1
  inside, so the out-of-range rule (gap in both starts) is not met. **Verdict: no measurable quality change**; the HC
  fusion (installed in prod the same day on the user's go) stands.
- TTFT reproduces the A/B above (−6…−7 % at 8k, −6.6…−7.1 % at 30k).

### 5ac. Validation of the prod config with HC fusion + F4 (prodval3): all paths taken, TTFT −5.4 / −6.2 %, outputs unchanged

User: "yes, both goes to prod". Prod's venv, launcher and drop-ins incl. 55 (`FN_HCFUSE=1`,
`FN_CG_MODE=FULL_AND_PIECEWISE`), default KV size, one start (as prodval2, §5y). 15 required path lines present
(incl. `FNHCFUSE fused hyper-connection kernels ran`, `Capturing CUDA graphs (FULL)`, the MTP name prefilter).
Hypothesis `tools/prodval3/HYPOTHESIS.md`; data `data/prodval3/`.

| | prodval2 (09-28 morning config) | prodval3 (+ HC fusion + F4) | Δ |
|---|---|---|---|
| code c=1 greedy / sampled, ms/tok | 14.74 / 16.60 | 14.60 / 16.37 | −0.9 % / −1.4 % |
| prose c=1, ms/tok | 23.66 | 23.49 | −0.7 % |
| code c=4, tok/s | 144.8 | 145.7 | ≈ |
| TTFT 8k / 30k (cold, nonce) | 2.886 / 10.39 s | **2.729 / 9.750 s** | **−5.4 % / −6.2 %** |
| greedy hashes (code / prose) | d102a738 / 38c70791 | identical | |
| start to ready | 2 min 39 s | 2 min 43 s | |
| KV pool at the default size | 840,265 tokens | 791,113 tokens | −5.9 % (full-graph memory) |

- All inside the hypothesis ranges. One start per config, so a validation, not an A/B; the A/Bs are §5z and §5aa.
- **New cost:** full CUDA graphs take memory, so the default KV pool shrinks 5.9 % (still ~790k tokens, far above
  what 16 streams at 32k use).

### 5ad. README re-measure on the full 09-28 prod config (bf16 state, pmu 64, fast loading, tool guards, HC fusion, F4)

User: "check if values in our README still apply after all patches, if needed remeasure". Prod's venv, launcher and
drop-ins, KV 4 GiB, 2 starts; one server per start runs nvprobe → concprobe → agentloop2 → turnreplay
(`tools/readme0928/`, hypothesis with per-row ranges written first). Data `data/readme0928/`.

| row | before (README) | 09-28 prod config, s1 / s2 | vs hypothesis |
|---|---|---|---|
| code c=1 greedy | 65.7 tok/s (09-27) | 14.675 / 14.627 ms/tok = **68.1 / 68.4 tok/s** (accept 4.37) | in |
| code c=1 sampled | 62.4 tok/s | 16.993 / 16.295 ms/tok = 58.8 / 61.4 tok/s | s1 out (slower) |
| prose c=1 greedy | 41.1 tok/s | 23.678 / 23.573 ms/tok = **42.2 / 42.4 tok/s** (accept 2.86) | in |
| code c=4 | 135.0–138.3 tok/s | **144.67 / 144.31 tok/s** | in |
| TTFT 8k / 30k (cold, nonce) | 2.77–2.78 / 10.25–10.33 s | **2.711, 9.694 / 2.722, 9.733 s** | in (8k at the edge) |
| 8k replay, first / cache hit | — | 4.166, 1.520 / 4.202, 1.516 s | — |
| agent loop (8 dependent turns) | 1.10 s/turn (09-26, K=3) | **1.25 / 1.25 s/turn** (46.4 / 46.6 ms/tok, accept 2.91) | **out (slower)** |
| warm turns, 46 SWE replays, median / mean | 0.57 s | **0.567, 0.615 / 0.563, 0.613 s** | in |
| 20k cached + 16 / 256 / 1,024 new | — | 0.678, 0.859, 1.468 / 0.673, 0.850, 1.456 s | — |
| prose c=4 / 8 / 16 aggregate | "~100 / 110 at 16 / 32" (old stack) | 83.6, 123.1, **161.9** / 80.8, 122.2, **165.0** tok/s | c=16 **out (below 180–260)** |
| KV in 4 GiB | 105,325 tokens | 105,325 tokens | confirmed |

- **Agent loop slower than the old row:** that row was K=3 (09-26). The loop's turns are short prose-like answers
  (accept 2.91), where K=5 is known to be ~5 % worse than K=3 (§5l); K=5 was chosen for code. Not attributed further.
- **c=16:** 162–165 tok/s, well above the old stack's ~100 but below my guess. c=32 cannot be measured on this config
  (`--max-num-seqs 16`).
- Greedy code hash d102a738 and the turn replay's final hash a70f8e4e reproduce the earlier runs of this config.

**§5v addendum (2026-09-28, user: "check GPU-side early exit for drafting") — what an implementable early exit is worth.**
Same logged drafts (`data/draftlog/`, 655 code / 992 prose verify steps), same cost model (base 43 ms, draft step 1.3 ms,
verify row 3.3 ms). The in-round stop (+4.1 % code / +7.5 % prose) assumes a stopped position saves its draft step
*and* its verify row, but the verify shape is chosen on the host before the drafts are known. Priced by design:

| design | what a stopped position saves | code (best τ) | prose (best τ) |
|---|---|---|---|
| ceiling: in-round stop, draft step + verify row | 1.3 + 3.3 ms | +4.1 % (0.70) | +7.5 % (0.65) |
| **IF nodes only** (drafts skipped on the GPU, verify keeps 1+K rows) | 1.3 ms | **+0.25 %** (0.4) | **+0.59 %** (0.3) |
| IF nodes + one host sync before verify (verify shrinks to 1+d), sync 0.9 / 2.0 ms | 1.3 + 3.3 ms, −sync | +2.7 / +0.9 % (0.75) | +5.9 / +3.9 % (0.65) |
| IF nodes + stopped verify rows marked padding (MoE skip, no sync), row residual 0.8 / 1.5 ms | 1.3 + (3.3 − residual) ms | +2.7 / +1.5 % | +5.0 / +3.6 % (0.45) |
| lagged host decision (from last round's confidences) | — | negative | ≤ +1.9 % (earlier replay) |

- **Prerequisite now met:** since F4 (§5z), prod's drafter captures FULL decode graphs, one per draft step (the log's
  "Capturing decode CUDA graphs (FULL)"; the fused whole-loop graph is still unsupported by `QWEN4_EXP_EXP_QSA_STATE`).
  So IF nodes (torch 2.13 `begin_capture_to_if_node`) can wrap each step's graph body.
- **IF nodes alone are worthless** (+0.25 / +0.59 %): skipping draft steps loses the same tokens as the stop policy but
  saves only the cheap part. The value needs the verify rows gone: one sync, or padding rows whose MoE is skipped.
- **Realistic ceiling: +1.5…+2.7 % code, +3.6…+5.9 % prose decode**, for 150–250 LOC at medium-high risk (capture
  constraints, forced rejection of stopped positions in the rejection sampler, AV plumbing). Agent turns are
  TTFT-bound, so the agent-level effect is smaller still.
- **Two unknowns decide it, both measurable cheaply before any build:** (1) the cost of one host sync per cycle on the
  current FULL-graph stack (an env-gated no-op `synchronize()` before verify, A/B); (2) the residual cost of a verify
  row marked padding (AV with a forced per-request budget).

**§5v addendum 2 — the cost of one host sync per verify cycle (`syncprobe2`, `tools/syncprobe/`): ≤ ~0.8 ms, cheap.**
Env-gated `current_stream().synchronize()` right after `speculator.propose()` (where an early-exit design reads the
draft count), clone venv with prod's config (HC fusion, FULL_AND_PIECEWISE, K=5, bf16 state, pmu 64), KV 4 GiB,
2 starts per arm; the patch was installed and removed by the runner (0 markers left). Run 1 (`syncprobe`) was void on
my own contradictory void rule. Data `data/syncprobe/`.

| | sync, s1 / s2 | no sync, s1 / s2 | Δ |
|---|---|---|---|
| code c=1 greedy, ms/tok | 14.798 / 15.012 | 14.843 / 14.654 | −0.3…+2.4 % (sign flips: noise) |
| code c=1 sampled | 16.694 / 16.732 | 16.655 / 16.537 | +0.2…+1.2 % (sign holds) |
| prose c=1 | 23.772 / 23.740 | 23.745 / 23.816 | null |
| code c=4, tok/s | 133.98 / 140.59 | 134.82 / 132.89 | null (mixed) |
| hashes | identical | | |

- **At or below the hypothesis range** (+0.5…+1.5 ms per cycle predicted): the one cell with a stable sign gives
  +0.13…+0.8 ms per ~66 ms cycle; the others are inside noise.
- **Consequence for the early exit:** the one-sync design sits in the cheap regime of the replay, i.e. **up to
  +2.7 % code / +5.9 % prose decode** (τ 0.75 / 0.65), with no padded-row mechanism needed. It still needs IF-node
  draft steps (prerequisite met since F4), forced rejection of stopped positions, and a host-known verify size
  1 + d from the synced count. 150–250 LOC, medium-high risk; agent-level effect ~1 %. A build decision, not queued.

### 5ae. dealignai's abliterated checkpoint, carried over into our format (`qwen38-flash-next-mtpfp4-ablit`)

User: "there is dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 maybe requant some components to same as our current
model". An uncensored research artifact (refusal directions projected out of weights by the upstream author); the work
here is a format conversion only. Tools `tools/ablit/`; data `data/ablit/`; provenance in the checkpoint's `PROVENANCE.md`.

- **What they changed:** their files are RadixArk's layer-sharded mixed checkpoint (the same lineage as ours); 203 of
  208 files are byte-identical to our prod checkpoint and all 384 expert shards to RadixArk. A per-tensor diff of the
  3 differing shards against RadixArk (run read-only on the backup box, file hashes = HF's): **13 tensors, all BF16
  `self_attn.o_proj.weight`** — the 12 full-attention layers (3, 7, …, 47) and `mtp.layers.0`. No experts, GDN, PLE or
  embeddings touched.
- **Our format for those tensors:** the 12 main o_proj are FP8 E4M3 128×128 blockwise in our checkpoint (from
  lovedheart's FP8-mixed conversion); MTP o_proj is RadixArk's BF16 unchanged. `scale = amax · (1/448)` reproduces all
  12 FP8 weights and scales **bit-exactly** from RadixArk's BF16 (`amax / 448` misses on ~0.07 % of bytes), so the
  carried-over tensors are what the original conversion would have produced from the abliterated BF16.
- **Build:** hardlinks to `qwen38-flash-next-mtpfp4` except the 3 rewritten shards (~8 GB new); per-tensor diff against
  prod's checkpoint = exactly the 25 intended tensors (12 weights + 12 scales + 1 BF16). `SHA256SUMS` for every file.

**§5ae serve test (`ablitval`, one start, prod venv + launcher + drop-ins, model swapped, KV 4 GiB): loads with every
path line; kernels as fast as prod; acceptance lower.**

| | abliterated (1 start) | prod, §5ad (2 starts) |
|---|---|---|
| code c=1 greedy, ms/tok (accept) | 15.635 (4.124) | 14.627–14.675 (4.365) |
| prose c=1, ms/tok (accept) | 26.696 (2.525) | 23.573–23.678 (2.856) |
| code c=1 sampled, ms/tok (accept) | 16.467 (3.922) | 16.295–16.993 (~3.94) |
| code c=4, tok/s | 134.68 | 144.31–144.67 |
| TTFT 8k / 30k, s | 2.698 / 9.673 | 2.711–2.722 / 9.694–9.733 |

- **Per verify cycle the cost is unchanged** (code 15.635 × 4.124 = 64.5 ms vs 64.0; prose 67.4 ms in both), and prefill is
  identical: the carried-over tensors run exactly like prod's. All of the decode loss is **fewer accepted draft tokens**:
  −5.5 % on code, −11.6 % on prose — **out of the hypothesis range** (±3 %).
- Two candidate causes, not separated (no counterfactual run): the outputs differ from prod's (different hashes), so
  acceptance is measured on different text; and our MTP drafter predicts the original model while the target's
  hidden states now come from 12 edited attention layers. dealignai report ~2.4 accepted per step with their own
  (also edited) MTP head on SGLang. A same-text check (teacher-forced drafter agreement on fixed text, both targets)
  would separate the two.

**§5ae — served (2026-09-28, user: "start it on 8080 with the api key file").** `vllm-flashnext-ablit.service`
(`tools/ablit/`): prod's environment with the model swapped, on 127.0.0.1:8080 as `flashnext-abliterated`, API keys from
the shared key file via a tmpfs `--config` YAML (never on argv; 4 of 4 keys accepted, no key → 401). Port 8080 is what
Open WebUI (LAN only) and the Cloudflare tunnel (behind Cloudflare Access) point at. Not enabled at boot; prod's unit
conflicts with it.

**§5aa — HC fusion, item 1 of the xn-style list (user: "do all"): L2 row sub-chunking refuted; K2/K3 are GEMM-efficiency-bound.**
Running K1→K2→silu→K3 over row sub-chunks (S = 256…1024) so the residual stays in L2 between kernels
(`tools/hcfuse/l2chunk.py`): **slower at every S** (M = 3,456: 1.90–2.09 vs 1.78 ms one pass; M = 1,024: 0.58–0.64 vs 0.56),
outputs within 1 ulp, reproducible. Out of the hypothesis range (−0.3…−0.6 ms). Per-kernel times (`tools/hcfuse/parts.py`,
M = 3,456): K1 0.707 ms for 159 MB = **225 GB/s (DRAM-bound)**; K2 0.475 ms for 78 MB = 164 GB/s; K3 0.579 ms for 95 MB =
164 GB/s; silu 0.017 ms. K2 and K3 each do ~23 GFLOP (~50 TFLOPS effective), so they are bound by GEMM efficiency (tile,
warps, stages), not DRAM: keeping their input in L2 cannot help, and the extra launches/partials cost more. The lever
for item 1 is a tile/config sweep of K2 and K3 (≈1.05 ms of the op's 1.78 ms).

**§5aa — FUSION-AGENDA item 1 closed: K2/K3 tile sweep (`tools/hcfuse/tune.py`) — no config breaks ~50 TFLOPS at the
prefill chunk size.** 152 K2 configs (BM/BN/BK/warps/stages × split-K 1/2/4/8) and 66 K3 configs passed the checks
(within 2 bf16 ulp of the current path, reproducible). Best vs current (ms):

| M | K2 | K3 | K2 + K3 saved |
|---|---|---|---|
| 3,456 | 0.471 → 0.471 (current is best) | 0.554 → 0.514 (−7 %) | −0.04 |
| 1,024 | 0.174 → 0.137 | 0.206 → 0.179 | −0.06 |
| 512 | 0.082 → 0.079 | 0.097 → 0.066 | −0.03 |
| 256 | 0.059 → 0.049 | 0.056 → 0.040 | −0.03 |
| 128 | 0.040 → 0.034 | 0.035 → 0.027 | −0.01 |

- Out of the hypothesis range (predicted −0.25…−0.45 ms at 3,456): the Triton GEMMs sit at their efficiency ceiling for
  these shapes. Summed over a 7.5k prefill ≈ −11 ms (~0.4 % TTFT), per agent turn ≈ −3 ms (~0.5 %): below a 2-start A/B's
  resolution, so not measured end-to-end and not proposed separately; the per-M table (`data/hcfuse/hctune.json`) can ride
  along with any future HC-fusion update.

### 5af. FUSION-AGENDA item 2: GDN output norm + FP8 quant in one kernel — 1.8× faster standalone, drift-level

The GDN output chain today: FLA `rmsnorm_fn` (gated, `norm_before_gate`, head dim 128, tile [4,128], 1 warp) writes bf16
`y` → `per_token_group_quant_fp8` (CUDA kernel, group 128, column-major scales) → `CutlassFp8BlockScaledMMKernel`
(out_proj). vLLM's norm+quant fusion pass matches `RMSNormGated` only on ROCm/AITER, so on CUDA these run separately.
Head dim = quant group = 128, so one Triton program row = one (token, head) = one quant group (`tools/gdnnq/gdnnq.py`).

| tokens | today (norm + quant) | fused | Δ |
|---|---|---|---|
| 3,456 | 0.861–0.871 ms | 0.474–0.481 ms | **−0.39 ms (−45 %)** |
| 595 | 0.097–0.109 | 0.026–0.036 | −0.07 |
| 128 | 0.041–0.042 | 0.018 | −0.024 |

- **Numerics: drift-level, not bit-identical.** The CUDA quant uses exact division (a fast-math `/` variant gave 16k
  mismatches; `div_rn` gives the rest identical). The residual: inside the fused kernel 57–59 of 21.2 M `y` elements
  differ from FLA's by one bf16 ulp, although the same norm code in a standalone kernel is bit-identical (0 of 21.2 M,
  with implicit, explicit, rtne and manual-RNE conversions alike) and FP contraction on/off changes nothing: the compiler
  schedules the combined kernel differently. Result: ≤ 21 FP8 bytes of 21.2 M and a few scales differ at 3,456 tokens;
  595 and 128 tokens bit-identical in the first run. A `noinline` norm helper would isolate it but Triton rejects
  constexpr parameters there. Accepted as drift (quality rule: no *noticeable* loss).
- In the hypothesis range for time (−0.26…−0.41 ms per GDN layer-chunk), out of it for bit-identity. Projected: ~−31 ms per
  7.5k prefill (36 GDN layers × 3 chunks, ~1.1 % TTFT); ~−2.5 ms per agent turn. Next: env-gated overlay on the clone venv
  (custom op, calls the same CUTLASS GEMM with the fused A/scales), op test, TTFT A/B.

**§5af server A/B (`gdnnq2`, clone venv, full prod config incl. HC fusion + F4, 2 starts per arm; run 1 `gdnnq` was void:
the eligibility check rejected the pass-through `_Fp8PbWoPartialBlock` scheme every FP8_PB_WO layer carries).**

| | gdnnq, s1 / s2 | base, s1 / s2 | Δ |
|---|---|---|---|
| TTFT 8k / 30k (median of 3) | 2.660, 9.489 / 2.656, 9.488 s | 2.709, 9.713 / 2.716, 9.735 s | **−1.8…−2.2 % / −2.3…−2.5 %** |
| code c=1 greedy: ms/tok (accept) | 14.231, 14.266 (4.474) | 14.567, 14.734 (4.365) | per verify cycle 63.67 / 63.83 vs 63.58 / 64.31 ms: **null** |
| code c=1 sampled: ms/tok (accept) | 14.282, 14.407 (4.436) | 16.414, 16.747 (3.941) | per cycle 63.35 / 63.91 vs 64.69 / 66.00: text-driven |
| prose c=1: per cycle | 67.31 / 67.50 ms (2.884) | 66.51 / 69.79 (2.856) | null |
| code c=4, tok/s | 138.9 / 142.54 | 141.3 / 139.13 | null |
| greedy hashes (code / prose) | 71fc9ede / 3bab2af1 | d102a738 / 38c70791 | **differ** (drift changes the greedy text) |

- **TTFT −1.8…−2.5 %**, above the hypothesis (−0.7…−1.5 %); prompts are identical and only the first token is timed.
- **Decode:** per verify cycle unchanged (the predicted ~−1 % is inside noise). The ms/tok differences (code −2.3…−3.4 %,
  sampled −13 %) come from *different outputs* being accepted differently (acceptance 4.47 vs 4.37, 4.44 vs 3.94), not
  from kernel time — not claimed. The replay's cold/warm times differ for the same reason (different output text).
- **The drift reaches the greedy text** on the probes (the op is ≤ 1 ulp on ~3e-6 of values, §5af standalone): a quality
  screen (GSM8K/HumanEval, as for the HC fusion) comes before any prod proposal.

**§5af quality screen (`evalgq`, §5r's GSM8K + HumanEval, greedy, 2 starts per arm, prod config + HC fusion + F4):
no measurable change.** The drift changes the text, not the answers:

| | gdnnq, s1 / s2 | base, s1 / s2 |
|---|---|---|
| GSM8K (1,319) | 96.21 / 95.83 % | 95.91 / 96.21 % |
| HumanEval (164) | 95.12 / 96.34 % | 95.12 / 95.12 % |

Every cross-arm McNemar p ≥ 0.33; the two base starts differ from each other (GSM8K 11 vs 15 discordant, Δ 0.30 pp) as
much as gdnnq differs from base (Δ −0.30…+0.38 pp). Per the quality rule (numeric drift is fine, no noticeable task
loss) GDNNQ is a **prod candidate: TTFT −1.8…−2.5 %, decode null, quality unchanged. Proposed, not installed** (the
user's call). Data `data/gdnnq/evalgq-score.txt`, `evalgq.txt`.

**FUSION-AGENDA item 4 closed (remaining FP8/FP4 activation quants into their producers): ~0.3 % TTFT, below resolution.**
Activation quantization in the 7.5k prefill window (ledger trace, `data/ledger/`): `per_token_group_quant` on the GDN
out_proj input 25.6 ms (0.96 %, **covered by item 2**), `cvt_fp16_to_fp4` on the MoE input 11.1 ms (0.41 %), the
2,560-dim FP8 quant of the attention/GDN inputs 7.6 ms (0.29 %), two small sites 1.4 ms (0.06 %). The producer of all the
remaining ones is our HC fusion's K3, whose bf16 output must still be written (router, shared expert and `in_proj_ba`
consume bf16), so a fused quant would save only the quant kernels' bf16 re-read, ≈ 0.3 % TTFT, below a 2-start A/B's
resolution, at the cost of four consumer hooks and the FP4 swizzled scale layout. Not built.

### 5ag. MoE prefill fusion in Triton (FUSION-AGENDA item 3, TODO 6): bit-identical to FlashInfer, TTFT −3.8…−4.3 % at 30k

User: "go for 3". Hypothesis and kill criteria written before any timing: `tools/moefuse/HYPOTHESIS.md`.

**Route.** The planned CUTLASS port (FlashInfer's SM120 FP8 dual-tile fused MoE → NVFP4) was replaced by Triton once a
probe showed that Triton 3.7.1 lowers `tl.dot_scaled(e2m1, e4m3 scales)` to the native
`mma…kind::mxf4nvf4.block_scale.scale_vec::4X` **when compiled for sm_120**. On sm_121 it falls back (MXFP4: bf16
upcast; NVFP4: a compiler assertion) because the sm_121 gate is Triton #10010, which 3.7.1 lacks. The wrapper sets
`triton.knobs.runtime.override_arch = "sm120"` around its own three launches only; Triton's in-memory cache key omits the
arch, so no other kernel is touched. sm_120 cubins run on sm_121.

**Design** (`tools/moefuse/moe_fp4.py`, prefill only, M ≥ 128 rows, decode stays on FlashInfer; FlashInfer's processed
tensors are read in place, no weight copy):
1. GEMM1 gathers A rows from the per-token FP4 input (no `expandInputRows`), computes the up and gate tiles in one CTA
   (w13 is `[up | gate]` after vLLM's reorder), then applies alpha → bf16 → `silu(gate)·up` → bf16 → NVFP4 with
   FlashInfer's fast-math recipe (`rcp.approx`, e4m3 scale, RN-even e2m1). No `doActivation`, no 88 MB bf16 round trip.
2. GEMM2 applies alpha2 and writes bf16 rows per (token, k).
3. A fixed-order finalize (k ascending, fp32), matching prod's non-fused DETFIN finalize.

**Correctness.** The quant recipe matches `ops.scaled_fp4_quant` bit for bit, codes and scales, at three global scales
with signed zeros. The whole MoE output is **bit-identical to FlashInfer's** at M = 3,456 / 595 / 128 (0 differing
elements, output buffer NaN-poisoned first so a no-op cannot pass) and bit-stable run to run.

**Tuning by bytes, not time.** The GPU was shared during tuning, so ncu L2-miss sectors (≈ DRAM reads, independent of
load) were the instrument. GEMM1's excess over the weight floor came from the gathered A rows: they were re-fetched for
every N tile after the weight stream evicted them. Making the N tiles of one M block adjacent cut GEMM1 from 1.25 to
**1.04 GB** (FlashInfer's GEMM1 reads 1.12 GB) and GEMM2 from 843 to 668 MB. A persistent GEMM1 and TMA weight loads were
both bit-identical and gave no gain. An intermediate byte probe was voided by Python late binding: the stored lambdas all
ran the last config.

**Standalone, quiet box, one layer, same process** (`test_moefuse.py bench`, 2 runs):

| M | FlashInfer | Triton | Δ |
|---|---|---|---|
| 3,456 | 9.64 / 9.72 ms | 8.39 / 8.43 ms | **−12.9…−13.3 %** (−1.25…−1.29 ms; predicted −0.7…−1.3) |
| 1,024 | 7.26 / 7.28 ms | 6.88 / 6.88 ms | −5.3…−5.5 % |
| 595 | 6.86 / 6.88 ms | 6.69 / 6.63 ms | −2.3…−3.6 % |

Per kernel at 3,456 (nsys minima): Triton GEMM1 + SwiGLU + FP4 4.33 ms (predicted 3.9–4.6, kill > 4.8), GEMM2 3.09 ms
(predicted 2.6–3.2), finalize 0.84 ms. FlashInfer: GEMM1 3.68 ms + `doActivation` 0.52 ms + `expandInputRows` 0.37 ms +
prefix sums and glue. The fused GEMM1 stage is 0.13 ms slower than FlashInfer's GEMM1 + activation; the win is the
removed expand/activation/glue passes and their bytes.

**Server A/B** (`armrun moefuse`, clone venv, full prod config incl. HC fusion + F4, `FN_MOEFUSE=1`, 2 starts per arm;
the path line `FNMOEFUSE Triton NVFP4 prefill MoE ran` is a required void check):

| | moefuse, s1 / s2 | base, s1 / s2 | Δ |
|---|---|---|---|
| TTFT 30k (median of 3) | 9.316 / 9.301 s | 9.720 / 9.671 s | **−3.8…−4.3 %** |
| TTFT 8k | 2.606 / 2.680 s | 2.709 / 2.697 s | −0.6…−3.9 % |
| replay cold / warm | 4.055, 4.155 / 1.520, 1.517 s | 4.168, 4.227 / 1.517, 1.522 s | cold −1.7…−4.0 %, warm null |
| code c=1 greedy, ms/tok | 14.621 / 15.017 | 14.907 / 14.805 | null (decode path unchanged) |
| prose c=1, ms/tok | 23.723 / 23.503 | 23.785 / 23.845 | null |
| greedy code / prose, sampled hashes | d102a738 / 38c70791, 245ccf19 | **identical** | outputs unchanged |

- **Against the hypothesis:** TTFT at 30k is inside the predicted −2.5…−5 %. The 8k prompt is partly below it: its last
  chunk is 591 tokens, where the kernel gains only ~3 %.
- **Quality:** the prefill MoE is bit-identical, and the greedy and sampled texts are identical across arms, so no
  quality screen is needed.
- **Status:** a prod candidate, **proposed, not installed**. It is installed env-gated (off unless `FN_MOEFUSE=1`) in the
  clone venv `vllm-venv-rssm` (`fused_moe/fn_moe_fp4.py`, hook in `experts/flashinfer_cutlass_moe.py`, backup
  `*.orig-moefuse`; `patch_moefuse.py <vllm> off` removes it). Data: `data/moefuse/`.

**§5ag addendum — both candidates together (`combo`: FN_MOEFUSE=1 + FN_GDNNQ=1 vs prod, 2 starts each, alternating;
hypothesis in `tools/moefuse/HYPOTHESIS.md`; user: "measure combination and compare to values in our README").**

| | combo, s1 / s2 | prod, s1 / s2 | Δ vs prod | README (09-28) | Δ vs README |
|---|---|---|---|---|---|
| TTFT 8k (7,503 tok) | 2.564 / 2.564 s | 2.727 / 2.729 s | **−6.0 %** | 2.71–2.72 s | −5.4…−5.7 % |
| TTFT 30k (29,263 tok) | 9.127 / 9.116 s | 9.725 / 9.757 s | **−6.1…−6.6 %** | 9.69–9.73 s | −5.8…−6.3 % |
| prefill rate 8k / 30k | ~2,930 / ~3,210 tok/s | ~2,750 / ~3,005 tok/s | | ~2,770 / ~3,020 | |
| code c=1 greedy, per verify cycle | 63.84 / 64.53 ms | 64.97 / 64.12 ms | null | | |
| greedy code / prose hashes | 71fc9ede / 3bab2af1 | d102a738 / 38c70791 | text differs (GDNNQ drift; = §5af's gdnnq arm) | | |

- **Against the hypothesis:** 30k 9.12 s is inside the predicted 9.00–9.25 s, and the gains add (MoE −4 %, GDNNQ
  −2.3 %). 8k 2.56 s is inside the predicted 2.55–2.66 s, at its fast end. Decode per verify cycle is null, as predicted.
- **ms/tok on code and sampled code moves (14.27–14.42 vs 14.69–14.89; sampled 14.5–14.6 vs 16.6–16.9):** that is
  GDNNQ's different output text accepting differently (4.474 vs 4.365 per cycle), as in §5af, not kernel time.
- **Open, not claimed: the cache-hit replay** (fixed 8k prompt, then 96 decoded tokens, twice) reads cold 4.40–4.42 vs
  4.21–4.28 s and warm 1.81 vs 1.52 s. The MoE-only run left it unchanged (§5ag), and the GDNNQ-only run showed the same
  1.81 s (§5af data), so it comes with GDNNQ. GDNNQ changes those 96 tokens, so this probe cannot separate text from
  kernel. A replay whose decoded tokens are forced (or zero decode, TTFT only) would settle it before a prod proposal of
  GDNNQ that relies on this row.

### 5ah. Deeper drafts need a confidence stop: measured cycle(K) + K=7 replay give +10 % code, +12.8 % prose over K=5 (not built)

User: "didnt we try drafting stop by confidence and longer possible drafts", then graph sizes 8−1 / 6−1 / 4−1, "distinct
probability limit for uneven draft token", "yes and also try additional capture graphs, i expect 3 to be hit often".
Hypothesis `tools/kstop/HYPOTHESIS.md` (before any run). Data `data/kstop/`.

**Cost of one verify cycle vs draft count** (`kcost2`: K = 2…7, each with its own exact FULL graph, prod config, KV 4 GiB,
2 starts each, nvprobe c=1; ms/tok × accepted per cycle):

| K (rows) | 2 (3) | 3 (4) | 4 (5) | 5 (6) prod | 6 (7) | 7 (8) |
|---|---|---|---|---|---|---|
| code, ms per cycle | 49.5 / 49.8 | 54.7 / 55.4 | 59.7 / 59.3 | 64.5 / 65.0 | 69.5 / 69.2 | 73.4 / 75.5 |
| prose, ms per cycle | 51.2 / 51.6 | 56.9 / 56.9 | 61.6 / 61.6 | 67.2 / 68.0 | 72.9 / 72.2 | 77.5 / 76.6 |
| code / prose ms/tok | 18.4 / 23.0 | 16.2 / **22.1** | 15.2 / 23.1 | 14.8 / 23.7 | **14.7** / 27.4 | 14.9 / 26.3 |

- **+4.8 ms per draft, near linear** (draft step + one verify row incl. the expert union); inside the predicted ranges.
  Prose cycles are 2–4 ms dearer. The old replay's 43 + K·4.6 ms was close but a bit cheap per draft.
- **Without a stop, depth does not pay:** code K5/K6/K7 within 1 %; prose best at K3, K6/K7 −12…−17 % (as predicted).
- **First run `kcost` void at K=4** (`expanded size 72 must match 75`): MRV2 rounds a capture size up to a multiple of
  1+K and a capture buffer disagrees when the rounded size is not itself in the list; prod's list is closed under
  rounding for K=5 only. `kcost2` uses per-K lists (1, 2, 4 + multiples of 1+K). Its K=2/K=3 starts of the first run
  were also CPU-contended by an unrelated pytest run and are discarded.
- **dynsd (FULL graphs for 3…8 rows in one server via MRV2 dynamic speculative decoding): blocked by a vLLM bug.** The
  drafter's graph manager derives its step length as `decode_query_len − num_speculative_tokens` and divides by it
  (`cudagraph_utils.py:314`, ZeroDivisionError) for a multi-step MTP drafter. Needs a small fix before any
  "graph per draft count" build.

**Replay** (`tools/kstop/replay2.py`, K=7 draft log `dl7`: 567 code + 950 prose verify steps, PIECEWISE because the
logging hook does not run inside FULL-graph replays; cost = the measured cycle(K) above). Validated: its fixed-K
predictions match the server (code K7 vs K5 +0.5 % predicted, ≈ 0 measured; prose −9.2 % vs −10 %).

| vs prod K=5, best τ | code | prose |
|---|---|---|
| stop, max depth 5, exact graphs | +4.4 % (τ 0.70) | +12.2 % (0.70) |
| stop, max depth 6, exact graphs | +8.0 % (0.70) | +12.6 % (0.75) |
| **stop, max depth 7, exact graphs** | **+10.0 %** (0.80; 4.4 drafts per cycle) | **+12.8 %** (0.80; 2.3) |
| depth 7, graphs for even row counts only, padded row 0 / 1.1 / 3.3 ms | +10.0 / +9.6 / +8.8 % | +12.8 / +12.0 / +10.4 % |
| depth 7, stop rounded up to fill the paid row | +9.3 % | +11.9 % |
| depth 7, two thresholds (τ_new / τ_padded), padded row 1.1 / 3.3 ms | +10.2 / +9.1 % | +12.1 / +10.9 % |

- **Depth + stop is the first-order lever; graph granularity is second order.** Exact graphs for every draft count beat
  even-only graphs by 0.4–1.2 points (code) and 0.8–2.4 (prose), depending on the unmeasured padded-row cost. The
  user's two-threshold rule recovers about half of that gap when padding is cheap. Prose stops early (2.3 drafts), so
  the 3-row graph the user expected to be hit often is exactly where prose lives.
- **Against the hypothesis:** code +10.0 % beats the predicted +4…+7 %; prose +12.8 % is inside +8…+14 %.
- **Not in the numbers:** one host sync per cycle to size the verify (≤ 0.8 ms, §5v addendum 2, ≈ −1.2 %); steps are
  treated as independent (greedy text is the target's, so only step boundaries move). Realistic: **code ~+8…9 %,
  prose ~+10…11.5 % single-stream decode**, for the build: IF nodes around the draft steps, one sync, a FULL graph per
  draft count (fix the dynamic-SD bug or capture our own), and the stop in the drafter.
- Decode only; agent turns are TTFT-bound, so the agent-level effect is smaller.

**§5ah addendum — is τ 0.8 optimal, and does verifying two candidates at ~50/50 pay?** (user; `tools/kstop/tau_cv.py`,
`data/kstop/tau_cv.json`, depth 7, exact graphs, measured costs, extra verify row 3.5 ms)
- **τ curve is flat-topped:** code 9.5–10.0 % for τ 0.70–0.90 (7.6 % at 0.95, 6.8 % at 0.50); prose 12.6–12.8 % for
  0.70–0.80 (11.7 % at 0.85, 7.7 % at 0.50). Too low a τ drafts deep into misses and costs more than it wins.
- **Cross-validated by prompt** (fit τ on 2 of the 4 prompts, score the other 2, all 6 splits): fitted τ 0.70–0.85;
  held-out gain at fixed τ 0.8 is code 8.9–11.0 % (mean 9.9), prose 10.3–15.4 % (mean 12.8), and 0.8 did at least as
  well as the in-sample fit in every split. So 0.8 is a robust choice **on this log**; not validated: sampled output,
  other prompt kinds (tool calls, long agent turns), and a real build (replay only).
- **Two candidates at the stop position** (one extra row for the drafter's second choice when p2 ≥ θ): the second
  choice is right at 40.5 % (code) / 36.4 % (prose) of breaks, and at 58.9 % / 49.5 % of the near-50/50 breaks
  (p1 0.35–0.65, p2 ≥ 0.25). A rescued break still gains exactly one token for one extra row, so: **code 0**
  (+9.92…+10.01 % vs +10.01 % without), **prose +0.9 points** at θ 0.15 (13.74 vs 12.83 %). With the stop, the ~50/50
  steps are mostly the ones already cut, so branching adds little; an add-on at most, not a lever of its own (§5v's
  earlier "branch: rejected" holds for code).

**§5ah correction (2026-09-29): the replay under-charged a stopped cycle.** To know draft j's confidence the drafter
must run step j, so a stop at j still pays that forward pass; it saves only the later steps and the verify rows. The
replay priced a stop at j as cycle(j). Re-priced as cycle(j) + one draft step (1.3–1.8 ms) on stopped cycles:
**code +8.4…+8.8 %, prose +9.4…+10.3 %** at depth 7 (best τ 0.70–0.75), not +10.0 / +12.8 %. After the ≤ 0.8 ms sync:
**code ~+7 %, prose ~+8…9 %**. The ranking of the designs does not change.

**§5ah build, phase 1 — FULL verify graphs for every draft count in one server: works** (user: "yes" to the build).
vllm#58821 (open; one-line guard for #58692, the dynamic-SD ZeroDivisionError) cherry-picked into the clone venv as a
removable overlay (`tools/kstop/patch_58821.py`, backup `*.orig-58821`). `dynsd2`: schedule 1 req → K7, 2 → K6, 3 → K5,
4 → K4, 5 → K3, 6–16 → K2 (`serve-dynsd.diff`, a runner copy of the launcher). 57 FULL graphs captured in 8 s, 1.84 GiB
(static K=7: 4 s, 0.81 GiB). At c=1 (K=7) greedy code / prose and sampled hashes are **identical to the static K=7
server**, code 14.82 vs 14.88 / 15.31 ms/tok, TTFT equal. At c=4 the scheduler runs K=4 (5-row verify graphs) cleanly:
134.2 tok/s, 3.88 accepted per cycle, no errors, so the RecoverSSM + MTP verify handles query lengths below its
maximum (spec_query_len 8). Data `data/kstop/armrun-dynsd2.jsonl`. Next: phase 2, a per-step draft count chosen by
the drafter (confidence) instead of by batch size.

**§5ah build — async scheduling costs us nothing to give up** (`noasync`, prod config K=5, 2 starts each; hypothesis in
`tools/kstop/HYPOTHESIS.md`). An exactly sized verify needs the scheduler to know each request's draft count, which the
AsyncScheduler (on by default: MTP is an EAGLE type) cannot. `--no-async-scheduling` (witness: `'async_scheduling':
False` in the non-default args):

| | no-async, s1 / s2 | async (prod default), s1 / s2 |
|---|---|---|
| code c=1, ms/tok (per cycle) | 14.568 / 14.671 (63.6 / 64.0) | 14.850 / 14.770 (64.8 / 64.5) |
| code sampled / prose, ms/tok | 16.58 / 16.51, 23.39 / 23.41 | 16.66 / 16.52, 23.78 / 23.26 |
| code c=4, tok/s | 147.7 / 148.1 | 132.0 / 132.8 |
| TTFT 8k / 30k | 2.699 / 9.648, 2.711 / 9.718 | 2.722 / 9.727, 2.714 / 9.700 |
| replay warm | 1.472 / 1.478 s | 1.519 / 1.521 s |

- **Outside the hypothesis** (predicted +2…+6 % cost): no cost; c=1 −0.6…−1.9 %, hashes identical in all four starts.
  With ~64 ms GPU cycles there is nothing for async scheduling to hide.
- **c=4 +11…+12 %** in this run, but the async arm sat below its usual 143–145 tok/s (§5ad), so not claimed until
  repeated. The launcher's "never combine MTP with async scheduling" caution is stale: the n-gram context now reads GPU
  token ids (`nvidia/model_state.py::_prepare_ngram_context`), not a CPU mirror.
- **Consequence:** the exact-size design (a verify of 1 + d rows) is free on the scheduling side.

**§5ah build — FNKSTOP overlay** (`tools/kstop/fn_kstop.py`, `patch_kstop.py`, env-gated `FN_KSTOP=1`, τ `FN_KSTOP_TAU`):
per draft step on the GPU a request stays active while the drafter's top-1 probability >= τ; the drafter's FULL
decode-step graphs are captured inside a CUDA-graph IF node on "any request active", so later steps are skipped on
the GPU; d_max goes to the host once per step and the scheduler gets `[-1] * d_max` drafts, so the next verify is
1 + d_max rows (a FULL graph per draft count via the dynamic-SD schedule + #58821). Unit test on the GPU
(`test_kstop.py`): 300 random confidence sequences, IF-node skipping, d per request, d_max and graph-padding masking
all match a Python reference, **0 mismatches**.

**§5ah build — first server runs.** Two voids before the first valid one: (1) capture failed with "operation would
result in a merge of separate capture sequences": the drafter's MoE shared experts fork to an aux stream, which a
CUDA-graph conditional body forbids; fixed by running them sequentially only while a draft step is captured inside the
IF node (`fn_kstop.IF_CAPTURE`, one check in `shared_experts.py`); (2) my own void rule required a log line that is
only written after the first request. `kstop3` (sched mode: the scheduler learns d_max, `--no-async-scheduling`):
draft-count histogram over 2,000 cycles {1: 465, 2: 287, 3: 228, 4: 177, 5: 120, 6: 124, 7: 599}; accepted tokens per
cycle match the replay (code 4.78 vs 4.56, prose 2.44 vs 2.46) but the cycle is 2.4–4.3 ms longer than modelled.

**Overhead diagnosis** (`kdiag`, 1 start per arm, all no-async; ms per verify cycle):

| | code | prose |
|---|---|---|
| static K7 / kstop τ 0 (never stops) | 73.8 / 76.1 (+2.3) | 76.1 / 80.1 (+4.0) |
| static K1 / kstop τ 1.1 (always d = 1) | 43.8 / 47.5 (+3.7, of it 1.3 the paid probe step) | 45.2 / 50.7 (+5.6) |

Skipped steps are cheap (H2 rejected); the ~2.3–4 ms is present with no stop at all (H1 exceeded). Mechanism: without
sizing, the CPU schedules the next step while the GPU is still drafting; the sizing sync made the CPU wait for the
drafts before scheduling. **Runner mode** (`FN_KSTOP_SIZE=runner`): the scheduler keeps K placeholders and does not
wait; the runner trims every decode request to d_max in `gather_batch_req_state`, right before the verify batch (the
scheduler counts the cut drafts as rejected, so its accounting holds). `krun`, 1 start each:

| | code ms/tok (per cycle) | sampled code | prose | c=4 tok/s |
|---|---|---|---|---|
| kstop runner, no-async | 13.84 (66.1) | 15.46 | 23.71 | 147.7 |
| kstop runner, async | 13.79 (65.9) | 15.44 | 23.14 | 136.2 |
| kstop sched, no-async (`kstop3`) | 14.26 (68.1) | 15.78 | 23.58 | 147.0 |
| prod K5 async (`noasync` as5, 2 starts) | 14.77–14.85 | 16.52–16.66 | 23.26–23.78 | 132.0–132.8 |

Runner mode recovers ~2 ms per cycle and works with async scheduling on. Greedy text is identical across all kstop
runs and differs from K5 (RecoverSSM commit grouping, §5b): a quality screen is required before any prod proposal.

**§5ah A/B (`kstopab`): confidence stop vs prod K=5, 2 starts each, alternating, clone venv, full prod config (async
scheduling on in both).** kstop = runner-mode sizing, τ 0.75, depth 7, dynamic-SD graphs (schedule bs1→7, 2→6, 3→5,
4→4, 5→3, 6→2, 7–16→1), #58821 cherry-pick.

| | kstop, s1 / s2 | prod K5, s1 / s2 | Δ |
|---|---|---|---|
| code c=1 greedy, ms/tok (accepted) | 13.738 / 13.714 (4.78) | 14.760 / 14.687 (4.37) | **−6.4…−7.1 %** (72.8 vs 67.9 tok/s) |
| code c=1 sampled, ms/tok | 15.473 / 15.379 | 16.662 / 16.454 | **−6.0…−7.6 %** |
| prose c=1, ms/tok (accepted) | 23.184 / 23.117 (2.44) | 23.916 / 23.484 (2.86) | **−1.3…−3.3 %** |
| code c=4, tok/s | 138.1 / 135.9 | 140.3 / 148.1 | −1.5…−8.3 % |
| TTFT 8k / 30k | 2.712, 9.664 / 2.711, 9.676 s | 2.745, 9.733 / 2.720, 9.757 s | −0.3…−1.2 % |
| replay cold / warm | 4.055, 1.418 / 4.073, 1.414 s | 4.181, 1.527 / 4.212, 1.513 s | cold −2.5…−3.3 %, warm −6.5…−7.4 % |
| greedy code / prose hashes | 89e8d183 / e18fc436 (both starts) | d102a738 / 38c70791 | text differs (RecoverSSM grouping) |

- **Inside the hypothesis on c=1** (code and sampled −5…−8 %, prose −3…+1 %). Below the replay's +8.4…10.3 % because
  ~1.3–2.7 ms per cycle of overhead remains (the d_max wait before the verify batch, sequential shared experts in the
  drafter's steps).
- **c=4 worse** (outside ±5 % on one start): this schedule gives batch size 4 at most K=4, below prod's K=5, and d_max
  is shared by the batch. The quality-screen schedule keeps K=7 up to 16 requests; a c=4 re-measure with it is the
  follow-up.
- Quality screen (GSM8K + HumanEval, 16 concurrent, schedule 1–16→K7) queued as `evalks`; control = this morning's
  `evalgq-base0/1` (identical config on the same venv; the overlays installed since are env-gated and inert).

**§5ah quality screen (`evalks`, GSM8K + HumanEval greedy, 16 concurrent, 2 starts; schedule 1–16 → K7 so the stop and
the d_max-sized verify are active at that concurrency; control = `evalgq-base0/1`): no measurable change.**

| | GSM8K | HumanEval |
|---|---|---|
| kstop, s1 / s2 | 95.91 / 96.66 % | 96.34 / 95.12 % |
| control (prod config), s1 / s2 | 95.91 / 96.21 % | 95.12 / 95.12 % |

Every kstop-vs-control McNemar p ≥ 0.087 (the lowest in kstop's favour); the two kstop starts differ from each other
more (GSM8K 4 vs 14 discordant, p = 0.031) than from the control: c=16 batch noise. Per the quality rule, the confidence
stop is a **prod candidate** (the user's call): code c=1 −6.4…−7.1 %, sampled −6.0…−7.6 %, prose −1.3…−3.3 %, TTFT
equal; open: c=4 with the 1–16 → K7 schedule (the A/B schedule dropped to K4 at batch 4 and lost 1.5–8.3 %). Data
`data/kstop/evalks-score.txt`.

**§5ah `kstopab2`: the prod-shaped schedule closes c=4** (bs1–10 → K7, 11 → 6 … 16 → 1; max-num-seqs 16; 2 starts each,
alternating, vs prod K5):

| | kstop, s1 / s2 | prod K5, s1 / s2 | Δ |
|---|---|---|---|
| code c=1, ms/tok | 13.724 / 13.733 | 14.887 / 14.714 | **−6.7…−7.8 %** (72.8 vs 67.6 tok/s) |
| code c=1 sampled, ms/tok | 15.362 / 15.411 | 16.592 / 16.474 | **−6.4…−7.4 %** |
| prose c=1, ms/tok | 23.116 / 23.175 | 23.272 / 23.646 | −0.4…−2.3 % |
| code c=4, tok/s (accepted per cycle) | 145.5 / 144.7 (5.04) | 146.5 / 145.3 (4.37–4.41) | −0.4…−1.2 % |
| TTFT 8k / 30k | 2.708, 9.701 / 2.693, 9.668 s | 2.713, 9.673 / 2.704, 9.704 s | equal |
| replay warm | 1.411 / 1.418 s | 1.512 / 1.520 s | −6.6…−7.2 % |

Inside the hypothesis (c=4 within ±3 %). **Complete prod candidate:** FNKSTOP runner mode, τ 0.75, FN_SPEC_N=7, this
schedule, #58821 cherry-pick, the launcher's FN_SPEC_DYN line; quality screen null (above).

**§5ah prod install — ON HOLD after validation (2026-09-29).** Installed (user: "integrate anything useful"): prod venv
FN58821 + FNKSTOP (backups `*.orig-58821`, `*.orig-kstop`), prod launcher FN_SPEC_DYN line (backup
`serve-flashnext.sh.orig-kstop`), drop-in `65-kstop.conf`. `prodval5` (prod's exact env incl. the §5ag fusions,
**default KV**, one start): all 20 path lines, TTFT 2.577 / 9.129 s, but code c=1 **14.163** ms/tok (accepted 4.53) vs
prodval4's 14.26 (−0.7 %, not −7 %), MiaAI quicksort greedy **52.2** vs 66.4 tok/s, chat sampled 39.6 vs 39.3, c=4
136.8 vs 138.8. Every A/B ran at KV 4 GiB; at the default KV the mapped PLE table pages, and K7 verifies up to 8 rows.
Drop-in 65 renamed to `65-kstop.conf.disabled` (prod = the validated fusions-only config; the venv code is inert
without FN_KSTOP). `ksprod` (prod venv, default KV, stop vs K5, 2 starts) queued to decide. Data
`data/prodval/armrun-prodval5.jsonl`.

**TODO 10a — block verification (`rejection_sample_method: block`, from the myllmbox v4 recipe): rejected.** `blockver`,
on the confidence-stop config (clone venv, KV 4 GiB), 2 starts each; hypothesis in `tools/kstop/HYPOTHESIS.md`.

| | block, s1 / s2 | strict (ours), s1 / s2 |
|---|---|---|
| code sampled, ms/tok (accepted per cycle) | 15.873 / 15.809 (3.96) | 15.352 / 15.345 (4.12) |
| code greedy / prose, ms/tok | 13.768 / 13.713, 23.053 / 23.209 | 13.720 / 13.760, 23.202 / 23.104 |
| c=4, tok/s | 145.2 / 139.4 | 145.8 / 144.1 |
| greedy hashes | 89e8d183 / e18fc436 (identical in all four) | |

- **Outside the hypothesis, the wrong way:** sampled code is **+3.0…+3.4 % slower**, with fewer drafts accepted per
  cycle (3.96 vs 4.12); greedy is unchanged and bit-identical, as predicted (the rule only differs on sampled drafts);
  c=4 −0.4…−3.3 %. Not adopted. Their recipe's acceptance of 5.1 of 6 is prompt-driven, not block verification.

**§5ah `ksprod` — in prod's exact config the confidence stop does not pay; drop-in 65 stays disabled.** Prod venv, prod
env incl. the §5ag fusions, default KV, stop vs K5, 2 starts each:

| | stop, s1 / s2 | K5, s1 / s2 | Δ |
|---|---|---|---|
| code c=1, ms/tok (accepted) | 14.175 / 14.220 (4.53) | 14.360 / 14.254 (4.47) | −0.2…−1.3 % |
| code sampled | 14.180 / 14.129 | 14.534 / 14.487 | −2.2…−2.8 % |
| prose | 23.464 / 23.531 | 23.525 / 23.251 | null…+1.2 % |
| code c=4, tok/s | 131.6 / 132.2 | 142.4 / 143.1 | **−7.6…−8.1 %** |
| MiaAI quicksort greedy, tok/s | 51.9 / 52.0 | 66.2 / 66.1 | **−21 %** |
| TTFT 8k / 30k | 2.56, 9.10–9.14 s | 2.55–2.78, 9.09–9.12 s | equal |

H1 of `ksprod` holds. The clone-venv gain came with +9 % accepted tokens per cycle (4.78 vs 4.37 on the no-GDNNQ text);
here the stop adds +1 % (4.53 vs 4.47) on GDNNQ's text, at default KV. Two differences, not yet separated: the text
(GDNNQ changes the greedy output and with it where the drafter is confident) and the KV size (the mapped PLE table
pages at default KV; K7 verifies up to 8 rows). A 2×2 (stop × KV 4 GiB / default, both with the fusions) would split
them. The venv code stays installed and inert.

**TODO 10e — vllm#58449 fused draft metadata (`f58449`, clone venv, K5, KV 4 GiB, armrun source toggle, 2 starts): bit-
identical, small.** Greedy hashes identical in all four starts (the metadata is the same, computed in place). Code c=1
14.750 / 14.764 vs 14.873 / 14.653 ms/tok (null); sampled null; **prose 23.666 / 23.657 vs 24.117 / 24.163 (−1.9…−2.1 %,
sign holds)**; c=4 132.4 / 142.8 vs 144.8 / 144.4 (one bad start); TTFT equal. Below the predicted −1…−4 % on code; a
+2 % prose effect with identical output. Not adopted on two starts; a candidate for a larger A/B, and its upstream PR is
the recipe author's (open). The venv was restored (0 markers).

**Agenda 4 — #58449 with 3 starts (`f58449b`, same spec): the prose signal was noise; dropped.** Witness line in all
three fused starts, absent in all rebuild starts; greedy hashes identical in all six. Prose fused vs rebuild per start
pair: 23.707 vs 23.513 (+0.8 %), 23.571 vs 23.499 (+0.3 %), 23.514 vs 23.954 (−1.8 %): the sign flips, so H0 holds
and `f58449`'s −2 % does not stand. Code c=1 14.73 / 14.65 / 14.72 vs 14.57 / 14.58 / 14.84 (null), c=4 147.0 / 141.6 /
134.5 vs 148.9 / 143.0 / 147.1, TTFT equal. Not adopted. Data `data/kstop/armrun-f58449b.jsonl`.

**§5ah `kssplit` — the prod loss is the text, not the KV size** (`ksprod` repeated at KV 4 GiB; prod venv, fusions on,
stop vs K5, 2 starts):

| | stop, s1 / s2 | K5, s1 / s2 | Δ (4 GiB) | Δ in `ksprod` (default KV) |
|---|---|---|---|---|
| code c=1, ms/tok (accepted) | 14.220 / 14.440 (4.53) | 14.375 / 14.435 (4.47) | −1.1…0 % | −0.2…−1.3 % |
| code sampled | 13.989 / 14.226 | 14.422 / 14.602 | −2.6…−3.0 % | −2.2…−2.8 % |
| prose | 23.304 / 23.453 | 23.118 / 23.827 | null | null |
| code c=4, tok/s | 151.2 / 151.2 | 141.0 / 141.7 | **+6.7…+7.2 %** | −7.6…−8.1 % |
| MiaAI quicksort greedy | 52.3 / 51.9 | 66.2 / 66.2 | **−21 %** | −21 % |

- **H-text holds for c=1:** flat at both KV sizes. On GDNNQ's greedy text the stop adds only +1 % accepted per cycle
  (4.53 vs 4.47; on the no-GDNNQ text it was +9 %, 4.78 vs 4.37), so the shorter-cycle gain disappears. The MiaAI
  prompt loses 21 % at both KV sizes: a text property too (agenda 2d explains it).
- **Only c=4 depends on the KV size:** +7 % at 4 GiB, −8 % at the default. Consistent with PLE paging under more rows
  per cycle at the default KV, not measured directly.
- Agenda 2e (concurrency ladder) is skipped: its condition, ≥ 4 % code c=1 gain at 4 GiB, is not met.

**TODO 10c — Marlin MoE (`--moe-backend marlin`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, the myllmbox setting) vs FlashInfer
CUTLASS (`marlin`, clone venv, K5, fusions off, KV 4 GiB, 2 starts): not adopted — run-to-run nondeterministic.**

| | Marlin, s1 / s2 | CUTLASS, s1 / s2 |
|---|---|---|
| greedy code / prose hashes | 23ff4843 / e0814028, **9211aa0b / f9038d55 (differ)** | d102a738 / 38c70791 (both) |
| code c=1 greedy, ms/tok (accepted) | 15.551 / 14.910 (4.06 / 4.26) | 14.529 / 14.807 (4.37) |
| code sampled | 15.556 / 14.262 | 16.443 / 16.803 |
| prose | 22.594 / 24.030 | 23.507 / 23.794 |
| code c=4, tok/s | **158.9 / 157.2** | 145.6 / 133.3 |
| TTFT 8k / 30k | 2.810, 10.146 / 2.819, 10.140 s | 2.726, 9.791 / 2.711, 9.701 s |
| replay cold / warm | 5.23, 2.27 / 4.48, 2.73 s | 4.20, 1.51 / 4.25, 1.52 s |

- **Greedy output changes between two starts of the same config**, which breaks the project's run-to-run determinism.
  Cause NOT established. It is not atomic add: this build hard-codes `use_atomic_add=False` in both Marlin MoE GEMMs
  (`experts/marlin_moe.py`; `VLLM_MARLIN_USE_ATOMIC_ADD` only reaches the dense Marlin path), the split-K reduce is
  fp32, the top-k combine is `ops.moe_sum`, and `VLLM_MOE_SKIP_PADDING` (the −1 top-k sentinel) was off. The 09-02
  bisection had already seen Marlin diverge at prefill (router logits bit-identical, `mlp.experts` first to differ).
  The c=1 cells therefore compare different texts (accepted per cycle 4.06 / 4.26 vs 4.37).
- **c=4 +8…+18 %** (Marlin's small-M GEMM beats CUTLASS FP4 at decode batches), **TTFT +3…+5 %** (bf16 math at
  prefill), warm replay +50…+80 % (partly text).
- Follow-up (agenda 5): first find the nondeterminism (standalone `fused_marlin_moe` on fixed inputs, twice per process
  and across two processes, with the real expert shapes); then Marlin at decode only, CUTLASS / the Triton prefill MoE
  (§5ag) at prefill; its weights are repacked, so it would need its own weight copy or a shared layout: scope first.
- **The Marlin MoE kernel itself is deterministic** (`tools/marlin/det_marlin_moe.py`, hypothesis there; Flash-Next
  expert shapes E=512, H=2560, I=640, top-k 10, random NVFP4 weights via vLLM's own helper): M = 1, 4, 55, 512 give one
  output class over 5 repeats, the same hashes in two separate processes, and bit-identical outputs with the tokens
  permuted — although `moe_align_block_size` builds a different per-expert token order on every call at M ≥ 55 (5 of 5
  orders distinct). So H-outside: the server divergence comes from the serving context, not the GEMMs. Candidates not
  yet tested: the weight repack at load, a shared Marlin workspace under CUDA graphs, and whether the Marlin arm is
  reproducible *within* one start (the probe records one hash per prompt per start, so within-start is unknown).
  Data `data/marlin/det_marlin_moe.json`.
- **In the server, Marlin is nondeterministic within one start, intermittently, with or without MTP** (`marlinrep2`,
  probe `tools/marlin/repro.py`: greedy A, A, B, A with a unique `cache_salt` each, then A twice as cache hits; 2 starts
  per arm; `marlinrep` start 0's Marlin arm matches). Hash of A per request:

  | arm | start 0 | start 1 |
  |---|---|---|
  | Marlin + MTP K5 | 1 class (62837025 ×6) | **3 classes**: salted repeats diverge at tokens 42 and 53; cache hits = A0 |
  | Marlin, no speculation | 1 class | **2 classes**: one salted repeat diverges at token 42 |
  | CUTLASS + MTP K5 | 1 class | 1 class |

  All arms' first A is the same text (62837025, Marlin and CUTLASS alike); B differs between Marlin starts, CUTLASS's B
  is one text. So H-run holds, not H-start: a runtime race in the serving context, independent of the drafter. The
  kernel is deterministic standalone, and Marlin's lock workspace is keyed per stream (`get_marlin_workspace`), so the
  remaining candidates are CUDA-graph replay of Marlin's per-call buffers, a stream overlap around the MoE call, or the
  moe_align order at shapes the standalone did not test. **Parked:** Marlin is not adopted; finding the race would
  mainly serve an upstream report (needs a go).

**Agenda 2f — batch draft count by expected value (`kstopval`, clone venv, K7 + stop τ 0.75 runner sizing, KV 4 GiB,
2 starts): not adopted, `max` stays.**

| | stop, max | stop, value | K5 |
|---|---|---|---|
| code c=1 greedy, ms/tok (accepted) | 13.697 / 13.796 (4.78) | 13.813 / 13.963 (4.78) | 14.758 / 14.713 (4.37) |
| code sampled | 15.421 / 15.378 | 15.540 / 15.605 | 16.488 / 16.483 |
| prose | 23.146 / 23.103 | 23.310 / 23.315 | 23.612 / 23.592 |
| code c=4, tok/s (accepted) | 137.97 / 148.07 (4.88 / 5.25) | 148.16 / 126.81 (4.66 / 3.95) | 145.64 / 145.13 |
| TTFT 8k / 30k | 2.71, 9.64 / 2.72, 9.72 s | 2.74, 9.74 / 2.71, 9.68 s | 2.75, 9.76 / 2.72, 9.75 s |

- c=1: identical greedy hashes max vs value in both starts (one request = the same rule, as designed); value costs
  +0.8…+1.2 % (the per-step value computation).
- c=4: H (value ≥ max by 2…6 %) is **not supported**: the sign flips between starts (+7 % then −14 %), and the
  stop arms' c=4 spread (±7 %) exceeds any effect. The shared draft count is not shown to limit c=4; value is dropped.
- The stop vs K5 (again): code c=1 −6.2…−7.2 %, sampled −6.5…−6.7 %, prose −1.9…−2.1 %, c=4 −5…+2 %, TTFT equal.
  Data `data/kstop/armrun-kstopval.jsonl`.

**Agenda 2c — no-async scheduling at c=4, re-verified (`noasync2`, clone venv, K5 prod config, KV 4 GiB, 3 starts
alternating): null; not proposed.**

| | no-async, s1 / s2 / s3 | async (default), s1 / s2 / s3 |
|---|---|---|
| code c=1, ms/tok | 14.741 / 14.653 / 14.646 | 14.619 / 14.796 / 14.568 |
| code sampled / prose | 16.75, 23.70 / 16.47, 23.26 / 16.50, 23.56 | 16.68, 23.54 / 16.63, 23.87 / 16.47, 23.64 |
| code c=4, tok/s | 142.8 / 149.2 / 146.6 | 148.2 / 146.1 / 145.5 |
| TTFT 8k / 30k | 2.73, 9.69 / 2.70, 9.71 / 2.71, 9.74 | 2.72, 9.71 / 2.72, 9.73 / 2.74, 9.72 |

- c=4 per start pair: −3.6 %, +2.1 %, +0.8 %: the sign flips, so H (+1…+3 % in every pair) fails and the bar for a
  proposal (≥ 3 % in every pair) is far off. c=1 flips too. Greedy hashes identical in all six starts.
- `noasync`'s +11…12 % is withdrawn: its async arm (132 tok/s) was an outlier low; async K5 reads 145–148 here.
  What stands from `noasync`: giving up async scheduling costs nothing at c=1.

**Agenda 2d — the confidence stop's −21 % on MiaAI's quicksort prompt (`miadiag`, clone venv, prod text: GDNNQ +
FNMOEFUSE, KV 4 GiB, stop vs K5, 1 start, probe `tools/kstop/miadiag.py`): it does not reproduce on the clone.**

| quicksort, greedy, 256 tokens | stop | K5 |
|---|---|---|
| tok/s (client side, median of 5) | **52.3** | **44.7** |
| tokens per cycle (accepted per cycle) | 2.70 (1.72) | 2.75 (1.76) |
| acceptance by position 0…6 | .70 .36 .22 .20 .11 .07 .06 | .74 .45 .29 .18 .10 |
| hash | cb1762a0 | b06b16cf |

- This is low-acceptance text (the prompt runs in the template's default thinking mode): 1.7 accepted per cycle. The
  stop drafts 1 token in 54 % of cycles (histogram after 475 cycles: d=1 255, 2 75, 3 31, 4 31, 5 11, 6 14, 7 58), so its
  cycles are cheaper at almost the same tokens per cycle: **+17 %, the stop working as designed.**
- The stop's rate matches the prod venv (51.9–52.3 in kssplit/ksprod); K5's does not (44.7 here vs 66.1–66.2 there). So
  the −21 % is K5 being fast on the prod venv for this one prompt, not the stop being slow. nvprobe's GDNNQ texts are
  the same on both venvs (K5 71fc9ede, stop d1a6e348), so the quicksort text is the open question: `miadiag2` (the
  same probe on the prod venv, kssplit config) is queued. Data `data/kstop/armrun-miadiag.jsonl`, `miadiag-stop0-hist.txt`.
- **`miadiag2` (prod venv, kssplit config, same probe, 1 start) closes it.** The stop reproduces exactly: text cb1762a0,
  52.5 tok/s, 2.70 tokens per cycle, the same acceptance by position. K5 writes a **different** reply here (a5fc09c4,
  the clone's K5 wrote b06b16cf) that drafts far better: 4.06 tokens per cycle (3.08 accepted; by position .87 .71 .62
  .49 .38) → 66.4 tok/s. **So the −21 % is K5 landing on an easy-to-draft reply on the prod venv, not a stop defect;
  on the clone, where K5's reply is hard to draft, the stop is +17 %.** This thinking-mode prompt is a near-tie text:
  the two K5 servers (prod venv + `serve-flashnext.sh` + capture list 1,2,4,6,8,… vs clone + `serve-dynsd.sh` + 1,2,4,6,12,…)
  write different replies although nvprobe's hashes agree. One 256-token reply ranks texts, not mechanisms; MiaAI's
  single-prompt number is not a basis for judging the stop. Data `data/kstop/armrun-miadiag2.jsonl`.

**Agenda 2b — GDNNQ warm replay (`gqreplay`, prod venv, prod env with FNMOEFUSE, KV 4 GiB, FN_GDNNQ on vs off, 2 starts,
probe `tools/gdnnq/replay1.py`): no regression, GDNNQ stays in drop-in 60.**

| | GDNNQ on, s1 / s2 | GDNNQ off, s1 / s2 |
|---|---|---|
| 8k prompt, 1 token: cold | 2.581 / 2.635 s | 2.708 / 2.637 s |
| same, warm (median of 3) | **0.122 / 0.123 s** | **0.128 / 0.129 s** |
| 8k + 96 tokens: cold / warm | 4.307, 1.763 / 4.316, 1.768 s | 4.463, 1.897 / 4.510, 1.896 s |

- H1 (warm slower with GDNNQ by > 5 %) is refuted: warm 1-token TTFT is 5 % *faster* with GDNNQ, and the 96-token warm
  replay 7 % faster (outputs equal to their own cold run in every cell).
- So the old 1.81 vs 1.52 s gap was never GDNNQ. It is a **venv/config gap**: the same probe shape reads 1.51–1.52 s on
  the clone venv (kstopab2, K5) and 1.76–1.90 s on the prod venv with either GDNNQ state. Not yet explained; the prod venv
  carries the inert FNKSTOP/FN58821 overlays and FNMOEFUSE (M ≥ 128 only, so not in a warm 96-token replay). Queued as
  agenda 2g.
- **Agenda 2g, closed (`venvtext`): there is no venv gap; my comparison was wrong.** The prod venv with FNMOEFUSE and
  without GDNNQ gives the clone's exact greedy hashes (code d102a738, prose 38c70791) and nvprobe's replay at
  **1.513 s**, code c=1 14.57 ms/tok. The 1.90 s came from `replay1.py`, whose prompt carries a different tag
  (`[replay-96tok]` vs nvprobe's `[replay-fixed]`), so it gets a different reply. The replay cell measures one 96-token
  reply's acceptance: it moves with K (K2 1.40, K3 1.58, K4 1.42, K5 1.52, K6 1.64 s) and with anything that changes
  the text (GDNNQ 1.76–1.82 s). Rank replay only between arms with identical text. Data `data/kstop/armrun-venvtext.jsonl`.

**Prod install of the §5ag candidates (user go, 2026-09-29) — validated, service still stopped.** Prod venv main1ea7
patched with FNMOEFUSE and FNGDNNQ (backups `*.orig-moefuse`, `*.orig-gdnnq`, scripts in
`/opt/llm/runners/prodinst0929`), drop-in `60-moefuse-gdnnq.conf`. `prodval4` (one transient start with prod's exact
environment, default KV, port 8092): all 17 path lines incl. `FNMOEFUSE Triton NVFP4 prefill MoE ran` and `FNGDNNQ
fused GDN norm+quant ran`; TTFT **2.567 s at 8k, 9.123 s at 30k** (combo A/B: 2.564 / 9.12); code 14.26 ms/tok, c=4
138.8 tok/s; greedy hashes 71fc9ede / 3bab2af1 (= the GDNNQ arms); replay warm 1.81 s (GDNNQ's text, §5ag open item).
Data `data/prodval/armrun-prodval4.jsonl`.


### 5aj. SWE-bench on TensorFold + EXL3 3.05 bpw: 52 / 58, inside (at the top of) our vLLM range

User: "can we run swt bench against tensorfold (tool calling)", then "do the 58 swt slice on tensorflow/exl3 (start
slow and watch)". Hypothesis in `tools/swe/HYPOTHESIS-swe.md` (written before the run: 40–50 expected).
Server: TensorFold 0.5.0 (EXL3 path stock) on `turboderp/Qwen3.8-Flash-Next-exl3` @ `69e33439` (3.05 bpw, 25/25 files
verified), `--context 65536 --max-tokens 16000 --parallel 4` (eager decode, MTP 1–6 drafts with its 30 % confidence
stop). Client exactly §5u's: `/root/fn-swe/run.sh`, `fn.yaml` byte-identical (1.0 / 0.95 / 20, reasoning_effort
medium, max_tokens 16000), mini-swe-agent 2.4.5 native `bash` tool calls, 4 workers, Verified-30 + Multilingual-28.
Both checkpoints ship the same chat template and generation config. Smoke (2 instances): 2 / 2.

| | Python /30 | Java/JS /28 | total /58 | gen s, Python / Java/JS |
|---|---|---|---|---|
| TensorFold + EXL3 3.05 bpw (run 1) | 27 | 25 | **52** | 2,152 / 3,357 |
| TensorFold + EXL3 3.05 bpw (run 2, a near-replay, see below) | 27 | 25 | **52** | 2,307 / 3,346 |
| vLLM today (prod config, NVFP4 `mtpfp4`, 65,536 context) | 27 | 24 | **51** | 2,619 / 3,219 |
| §5u vLLM prod (FP8 GDN), runs 1 / 2 | 24 / 28 | 24 / 24 | 48 / 52 | 2,345, 2,999 / 3,505, 3,323 |
| §5u NVFP4 GDN | 28 / 27 | 23 / 23 | 51 / 50 | 2,363, 2,725 / 3,929, 3,499 |
| §5u bf16 SSM state | 26 / 27 | 24 / 24 | 50 / 51 | 2,810, 2,773 / 2,982, 2,906 |

- **Quality: no loss at 3.05 bpw on this benchmark.** 52 equals the best §5u run; one run, so "inside the range", not
  "better". Per instance against each §5u arm: 1–3 solved there and not here, 3–7 the other way (TensorFold alone
  solved django-16667 and babel-15445 against every arm). All 58 trajectories ended `Submitted`: 0 context
  overflows (§5u had 1–3 per run), no empty patches.
- **Tool calls:** TensorFold's parser worked throughout (58 trajectories, no format-error loops); 0 server tracebacks.
- **Speed is not comparable to §5u:** those servers predate HC fusion, F4, the Triton prefill MoE + GDNNQ and ran at the
  default KV size. Python generated 8–28 % faster, Java/JS within §5u's range.
- **Same-day vLLM baseline (2026-09-30, `tools/swe/vllmswe2.sh.txt`):** today's prod config (`vllm-venv-main1ea7`,
  `qwen38-flash-next-mtpfp4`, MTP K=5 + probabilistic drafting, HC fusion, GDNNQ, MoE fusion, FULL_AND_PIECEWISE,
  `--max-model-len 65536`, KV 8 GiB, 16 seqs), same client and sampling. **51 / 58**, all 58 `Submitted`, **0 context
  overflows** (§5u, also at 64k, had 1–3 per run; with one run here that is within run-to-run variation, not a
  cause), so the user's rule (count an overflow as solved) changes nothing in either row today. TensorFold + EXL3 against it: 4 solved there only (babel-15445, django-16667, matplotlib-20859,
  lombok-3594), 3 here only (django-13551, django-14034, gson-2311): one instance apart, one run each, **a tie**.
  Generation time: TensorFold **18 % faster on Python** (2,152 vs 2,619 s), vLLM **4 % faster on Java/JS** (3,219 vs
  3,357 s); total 5,509 vs 5,838 s, TensorFold −6 %. Both at 4 workers, so this is c≈4 agent throughput, where
  TensorFold runs eager and vLLM with graphs.
- Draft acceptance over the run: 65–66 % (TensorFold's /health counters).
- **TensorFold run 2 is not an independent sample.** Same 52 resolved, and 48 of 58 patches byte-identical to run 1
  (Python 26/30, Java/JS 22/28); the first trajectory checked (astropy-14539) matches run 1 message for message up to
  message 26 of 31. Cause: without a `seed` in the request TensorFold seeds each reply from the prompt
  (`exact_sampling.seed_for`: sha256 of the prompt tokens), so the same conversation samples the same reply; runs part
  only where a tool output differs (timings, container state). mini-swe-agent sends no seed. vLLM without a seed draws
  a fresh one per request, so its two §5u runs are independent and TensorFold's two here are not: **TensorFold is one
  run on this slice, not two**, and its run-to-run spread is not measured. Fixed for the hard slice: `tfhard2` layers
  `tools/swe/fn-tfhard2.yaml` (`seed: 2` on every request) through `run2.sh`'s per-arm config; `tfhard1` stays
  prompt-seeded like run 1.
- **GSM8K + HumanEval screen on the same server (`evalprobe`, greedy, 16 concurrent; control `evalgq-base0/1` = vLLM
  prod on our checkpoint, 2 starts; `data/swe-tf/evalq-tfexl3-score.txt`): no loss.** TensorFold + EXL3 GSM8K
  1273/1319 = **96.51 %** (vLLM 95.91 / 96.21 %), HumanEval 159/164 = **96.95 %** (vLLM 95.12 / 95.12 %). Paired per
  item: GSM8K 16 vs 8 and 14 vs 10 in TensorFold's favour (McNemar p = 0.15, 0.54), HumanEval 3 vs 0 (p = 0.25); the
  two vLLM starts differ by 11 vs 15 among themselves. Nothing significant; the sign is EXL3's way on all four
  comparisons. Truncated at the probe's budget: GSM8K 16, HumanEval 2.
- **Verdict against `tools/swe/HYPOTHESIS-swe.md`:** the quality hypothesis (40–50 of 58, "a larger cut than any §5u
  arm") is **refuted in the good direction**: 52 / 58 on SWE (one effective run) and no loss on GSM8K/HumanEval.
  3.05-bit EXL3 experts do not cost measurable task quality against our NVFP4 experts on these benchmarks. Harder
  slices follow (§5al) because 88–90 % on the 58 leaves little headroom to separate the two.
- **Prefill is TensorFold's weak side:** the same probe's TTFT is 9.27 s at 8k and 36.3 s at 30k tokens, against 2.7 s
  and 9.4–9.9 s for vLLM prod in the same probe (`evalgq-base0/1`, 2.64 / 2.71 s at 8k), 3.4–3.9× slower. On agent loops that is partly hidden by 4-way concurrency.
Data `data/swe-tf/` (harness reports, result JSON).

### 5ak. The underwater-scene prompt on both engines (two seeds)

User: "after the run, let both models do: [the scene prompt]", then "is 32k output enough for this?" (no: default
thinking is xhigh, so the budget was raised to 100k output, 128k context). `tools/fishscene/`: `prompt.txt`, `gen.py`
(one request, seed 1, the model's default sampling and thinking, max_tokens 100000), `check.py` (mechanical checks of
the hard constraints). Servers: TensorFold + EXL3 3.05 bpw `--context 131072 --max-tokens 100000 --parallel 1`; vLLM
today's prod config at `--max-model-len 131072`. One request each, nothing else on the box.

| | output tokens (thinking) | s | tok/s | HTML bytes | < 18 KB | other checks |
|---|---|---|---|---|---|---|
| TensorFold + EXL3 3.05 bpw | 26,234 (17,987) | 374.7 | 70.0 | 19,910 | **no** | all pass |
| vLLM, our checkpoint | 23,980 (14,769) | 523.0 | 45.9 | 21,507 | **no** | all pass |

- Both finished (`stop`), start with `<!DOCTYPE html>`, no fences, no `<link>`/`<script src>`/fetch/URLs, no
  pictographs, both use a `linearGradient`. **Both break the one numeric constraint, the 18 KB limit** (18,432 bytes):
  TensorFold by 8 %, vLLM by 17 %.
- Static SVG counts differ (TensorFold 5 ellipses, 4 circles, 15 paths; vLLM 0 ellipses, 6 circles, 25 paths); bubbles
  may be created in script, so the counts say nothing about the required 5 bubbles. Fish count, eyes, wrap-around and
  motion need a look, not a regex: side by side at http://10.0.0.133:8765/ (LAN) and on the artifact page.
- Speed: single request, one sample each, so a sanity figure, not a benchmark. vLLM's 46 tok/s sits at its prose
  rate (23 ms/tok at c=1, #58863 table) since most of the reply is thinking; TensorFold's 70 tok/s on the same kind of
  text is unmeasured elsewhere, so this is one data point, not a ranking.
- **Round 2 (seed 2, user: "do both scenes once more", `mixswe2.sh.txt`):**

| | output tokens (thinking) | s | tok/s | HTML bytes | < 18 KB | other checks |
|---|---|---|---|---|---|---|
| TensorFold + EXL3 3.05 bpw | 30,427 (22,133) | 412.6 | 73.7 | 19,818 | **no** | all pass |
| vLLM, our checkpoint | 29,398 (20,357) | 631.5 | 46.6 | 22,084 | **no** | all pass |

- **What they look like** (headless Firefox 1280×800, frames at 0, 4 and 12 s; no other browser checked). Seed 1 is
  broken on both: TensorFold draws water, sand, seaweed and rising bubbles but **no fish** in any frame (the legend
  names two); vLLM draws **almost nothing** (no gradient, no floor, one fish clipped in the top-left corner, frozen).
  Seed 2 works on both, each with one deviation: TensorFold adds a school of ~12 small background fish silhouettes
  (against "exactly 2 fish"), vLLM shows no tan/brown seafloor. Every reply breaks the 18 KB limit.
- **Verdict:** four samples, no engine difference: each engine produced one broken and one nearly-correct scene, the
  same failure rate. The spread is between seeds, not engines, so nothing here ranks the EXL3 quant against NVFP4.
- Page with all four live: https://claude.ai/artifact/9KEoopw1XJzh7fyjiMxCXo (private); LAN http://10.0.0.133:8765/.
  Data `data/fishscene/` (html, json, `check-round{1,2}.jsonl`).

### 5al. Harder slices: TensorFold + EXL3 at least level (48/47 vs 47/41), and 64k context is too small

User: "should we add some harder swt cases" → "replace 10 easy with 5 harder and 5 hardest" → (Java/JS) "yes, prune
after current run and swap slices". Hypotheses written before the runs: `tools/swe/HYPOTHESIS-swe.md` "Hard slice" and
"Java/JS swap". **Mixed Python 30** = 20 kept + 10 hard Verified (all three ">4 hours" + seven "1-4 hours", seed 1;
`set-hard.txt`); **mixed Java/JS 28** = 18 kept + the 10 largest reference patches in the same repos (`set-jshard.txt`,
23–219 lines). The 20 dropped were easy instances every earlier run solved. Same servers, client and sampling as §5aj
(64k context, max_tokens 16000, 4 workers). Kept instances are scored from the earlier runs, so: vLLM run 1 = vllmnow1
(kept) + vllmhard1; vLLM run 2 = vllmmix2 (all 58 fresh); TensorFold run 1 = tfexl3r1 + tfhard1; TensorFold run 2 =
tfexl3r2 (kept: **a replay of run 1**, §5aj) + tfhard2 (`seed: 2`, independent; verified: first tool calls differ).
Overflows are **not** counted as solved here (the 128k reruns below settle them).

| | Python /30 | Java/JS /28 | **total /58** | hard 20 alone | context overflows | gen s, hard 20 |
|---|---|---|---|---|---|---|
| vLLM run 1 | 24 | 18 | **42** | 11 (7 + 4) | 6 | 4,424 |
| vLLM run 2 | 21 | 18 | **39** | 9 (6 + 3) | 3 | (full 58: 7,855) |
| TensorFold + EXL3 run 1 | 25 | 20 | **45** | 13 (8 + 5) | 3 | 3,918 |
| TensorFold + EXL3 run 2 | 23 | 19 | **42** | 10 (6 + 4) | 5 | 4,339 |

- **Engines: TensorFold + EXL3 is 3 ahead in both pairings (45 vs 42, 42 vs 39), +2 and +1 on the hard 20 alone** (the
  independent part). Paired on the hard 20: TensorFold-only 4 + 3, vLLM-only 2 + 2 (7 vs 4 over both runs; a sign test
  gives p ≈ 0.55). Inside the ±3 the hypothesis allowed, with the sign holding twice: **no quality loss from EXL3
  3.05 bpw on harder work either, and no demonstrated gain.**
- **Hard Python ran above the prediction** (2–6 expected; 7, 6, 8, 6): Verified's time labels track difficulty for
  this model only loosely (the easy halves score ~90 %, these ~70 %). Solved by all four runs include the ">4 hours"
  sphinx-7590; unsolved by all: xarray-6992 (">4 hours"), axios-5316, lombok-3371, vuejs-11739, vuejs-11899.
- **Java/JS hard inside the prediction** (3–7; 4, 3, 5, 4): the large-patch set is the harder half.
- **Overflows are the same condition on both engines**, reported differently: vLLM rejects prompt + max_tokens > 65,536
  (`ContextWindowExceededError`), TensorFold refuses the same ("... exceeding the server's 65536-token safe cache
  capacity", which mini-swe-agent files as `BadRequestError`). Every overflow is a hard instance at a prompt of
  ~49.5k+ tokens; vuejs-11739 and lombok-3371 overflow in all four runs. The first overflow-rerun set builder matched
  only vLLM's wording and found no TensorFold overflows; `make-ovf.sh` v2 + `is-ovf.py` match both (dry-run: vLLM sets
  unchanged, TensorFold 3 + 5), fixed before the chain reached TensorFold.
- Data `data/swe-tf/` (reports `openai__flashnext.FN_{vllmhard1,vllmmix2,tfhard1,tfhard2}_*.json`, `*-result.json`).

**Overflow reruns at 131,072 context (user: "re run failed with bigger context"; `finale.sh.txt`).** Every overflowed
trajectory re-run on the same engine, sampling and seed config at 131,072 (vLLM KV 12 GiB, MemAvailable 25 GiB at
ready; TensorFold `--parallel 4`, 18 GiB). **15 of 17 solved, none overflowed again**: vLLM 7 / 9 (run 1: 5 / 6, only
lombok-3215 failed; run 2: 2 / 3, lombok-3371 failed), TensorFold 8 / 8. lombok-3371 and vuejs-11739, which had
overflowed in all four 64k runs, were solved in 3 and 4 of their reruns. Against the hypothesis (25–50 %; 0–2
overflowing again): **refuted** — at this length the overflowed trajectories were nearly done, not stuck, so the
user's "with longer context it would pass" was right, and **64k is too small for this agent on hard work**.

**Final mixed table, overflows replaced by their 128k reruns:**

| | Python /30 | Java/JS /28 | **total /58** | hard 20 alone |
|---|---|---|---|---|
| vLLM run 1 | 26 | 21 | **47** | 16 |
| vLLM run 2 | 22 | 19 | **41** | 11 |
| TensorFold + EXL3 run 1 | 26 | 22 | **48** | 16 |
| TensorFold + EXL3 run 2 | 25 | 22 | **47** | 15 |

- **TensorFold + EXL3 ahead in both pairings again (+1, +6)**; paired discordances 5 vs 4 and 9 vs 3 (14 vs 7 over
  both, sign test p ≈ 0.19). Weaker than it looks: TensorFold run 2's kept 38 replay run 1, so babel-15445,
  django-16667 and matplotlib-20859 count in both pairings; the independent comparison is the hard 20 (16 vs 16,
  15 vs 11). **Verdict: EXL3 3.05 bpw on TensorFold is at least as good as our NVFP4 on vLLM on this benchmark**; the
  lead is consistent in sign but not significant at n = 58 × 2.
- vLLM run 2 (41) is the outlier, the one run where every instance was fresh; its spread against run 1 (47) is the
  run-to-run variation at temperature 1.0 on this slice, ±3 around 44.
- Unsolved by all four even at 128k: axios-5316, xarray-6992 (">4 hours"), vuejs-11899.
- Serving consequence for agent work: run at 131,072 (vLLM: KV 12 GiB fits with 25 GiB free; TensorFold: 4 slots fit
  with its table still locked). Data: `openai__flashnext.FN_{vllmhard1x,vllmmix2x,tfhard1x,tfhard2x}_*.json`,
  `*x-ovf*-result.json`.

### 5am. The pasture prompt: TensorFold 2 of 2 usable, vLLM 1 of 2

User: "at the end do two runs with: [pasture prompt]". Same harness as §5ak (`gen.py` with `GEN_PROMPT`, seeds 1 and
2, 100k output budget, the model's default sampling and thinking), servers at 131,072 context (vLLM from the overflow
reruns, KV 12 GiB; TensorFold the scene config, 1 slot). `check.py` with `CHECK_LIMIT_KB=30`; visual check in headless
Firefox at 1280×800, frames at 4 and 12 s. Hypothesis `tools/pasture/HYPOTHESIS.md`.

| | output tokens (thinking) | s | tok/s | bytes | < 30 KB | format | visual |
|---|---|---|---|---|---|---|---|
| vLLM, seed 1 | 34,466 (20,079) | 705.4 | 48.9 | 31,889 | **no** | ok | complete scene |
| vLLM, seed 2 | 46,959 (30,193) | 1004.2 | 46.8 | 39,526 | **no** | ok | **broken** |
| TensorFold + EXL3, seed 1 | 47,697 (33,315) | 682.8 | 69.9 | 30,432 | yes | ok | complete scene |
| TensorFold + EXL3, seed 2 | 53,585 (44,065) | 730.1 | 73.4 | 20,765 | yes | **fenced** | complete scene |

- **What renders.** vLLM seed 1: sky gradient, sun with halo and rays, grass, hedge, 4 animals (cow with patches,
  pink nose and udder; wool-circle sheep with a dark face; pink pig with snout; brown horse with mane and tail); the
  trees sit cut off at the frame edges and the sheep's and pig's legs are partly lost in the grass. vLLM seed 2:
  **dark ground, a pig clipped in the top-left corner and only the legend panel**, the same failure as §5ak's vLLM
  seed 1. TensorFold seed 1: the fullest scene (fence, three rooted trees, clouds, sun, all four animals with their
  features). TensorFold seed 2: a clean complete scene with labelled animals, but the reply is wrapped in a
  ```` ```html ```` fence, which the browser shows as a text line at the top (it breaks "starts with <!DOCTYPE html>").
  Leg counts and bounce phases were not counted frame by frame; "exactly 4 legs" is judged by eye from the stills.
- `check.py` flags vLLM seed 1 for a "pictograph": `№` in its title text ("FIELD STUDY № 04"), text, not a graphic.
- **Against the hypothesis:** 0–2 of 4 meeting everything → 1 (TensorFold seed 1; the fenced one misses only the
  format rule); 2–4 under 30 KB → 2 (both TensorFold); no engine difference separable at n = 2 → holds formally, but
  across both prompts vLLM produced the same "dark page, one shape clipped top-left" failure twice and TensorFold once
  a page without fish, so 2 of 4 vs 1 of 4 broken renders over the two prompts, still n too small to rank.
- TensorFold thinks longer here (33–44k thinking tokens vs 20–30k) and still finishes faster (70–73 vs 47–49 tok/s,
  one request each). Page: https://claude.ai/artifact/UX283L4b7bkk2S7ehSjnWM (companion to the fish page), LAN
  http://10.0.0.133:8765/. Data `data/pasture/`.

### 5an. TensorFold 0.6.0 on the Spark: exact, faster decode on EXL3, prefill 2.3–4.4× vLLM, and prompts queue oldest-first

User: "do the update and test". tf-venv moved to 0.6.0 (`c464617`, local branch `v060`); one server start per
checkpoint, `--context 262144 --parallel 8`, nothing else on the box. Hypothesis `tools/tf060/HYPOTHESIS.md`; data
`data/tf060/`. Our checkpoint's long mix was stopped at the user's "fast tests first" and is deferred.

| | EXL3 3.05 bpw | our `mtpfp4` (block FP8 + NVFP4) |
|---|---|---|
| startup estimate / budget | 61.97 / 104.36 GiB | 84.24 / 104.60 GiB |
| n-gram table | locked, **22.2 s** to lock (0.5.0: 0.3–0.4 s in six starts → T4) | not locked (no room), read in 12.2 s |
| exactness: concurrent == alone / alone == serial | **84/84, 30/30** | **84/84, 30/30** |
| decode c=1, `bench_openai` 400 tok, greedy fib / chat | 77.1 / 61.2 tok/s | 57.9 / 43.2 tok/s |
| decode c=4 aggregate, code greedy | 239.5 tok/s | 200.2 tok/s |
| TTFT 2k / 8k / 32k / 65k (`prefill_cold`) | 2.75 / 11.4 / 46.0 / 93.6 s | 1.47 / 6.0 / 25.1 / 51.3 s |
| prefill rate | flat **~720 tok/s** | ~1,280–1,395 tok/s |

- **Exactness holds on both** (hypothesis met).
- **Decode, our checkpoint vs 0.5.0 + our branch** (TF#126 table: fib 59.4 / chat 37.6 greedy): fib −2.5 %, chat
  +15 %. Outside the +3…+10 % hypothesis in both directions, and not a clean comparison: 0.6.0's `lm_head` on FP8G
  and bf16 prompts change the greedy text, so each cell decodes a different reply (§5ah: one reply ranks texts).
- **TTFT** (hypothesis: still 2.5–4× vLLM's 2.6 s at 8k): ours **2.3×** (6.0 s), EXL3 **4.4×** (11.4 s). Both
  rates are nearly flat with length, so a per-token cost dominates, not attention. EXL3 prompts never read the
  `--prefill-fp8` switch (only `cuda/nvfp4/linear.py` does), so bf16 prompts do not explain EXL3's rate; the
  0.5.0 figure (9.27 s at 8k) came from a different probe and is not comparable. Next: profile + chunk-size sweep.
- **Long mix, EXL3** (4 × ~120k-token prompts at t=0, 8 × 2k every 15 s): no errors, **no out-of-memory stops**,
  MemAvailable min **12.15 GiB** (gate reserve 2 GiB) — as hypothesised. But **the short requests waited 630–715 s
  for their first token** (hypothesis: decode at 15–35 tok/s alongside): CUDA fills prompts **oldest first**
  (`multi.py` `_pieces`), so each 2k request sat behind ~480k prompt tokens at ~660 tok/s; the live long streams
  decoded 0.7–3.9 tok/s while others filled (one round per prompt pass). TensorFold's Mac path already fills
  "fewest tokens left first"; the CUDA one does not → TODO **T9**. Major faults over the run: 15,092.

- **Prompt precision on our checkpoint** (user: "check this also"; FP8, bf16, FP8 starts after the §5an bf16 start,
  8k/32k × 3 each, medians): FP8 prompts **8k 5.19 / 5.25 s vs bf16 6.00 / 6.08 s; 32k 21.36 / 22.07 s vs 25.09 /
  25.50 s** → FP8 **1.14–1.19× faster** (each arm's starts within 1–3 %). Below the 1.2–1.5× hypothesised: only the
  dense projections switch (experts, attention and the GDN chain do not). Even FP8 prompts leave our checkpoint at
  ~2.0× vLLM's 8k TTFT. Accuracy cost not measured here (Flash Next on CUDA returns no logprobs, TF #108).

### 5ao. TensorFold n-gram work, first measurements: T4 closed, T2's warm cost nil, T1 helps prompts but costs decode, cold cells void

Hypotheses `tools/tfple/HYPOTHESIS.md`; data `data/tfple/`. Branch `ple-gather` (worktree `~/git/tensorfold-ple`).

- **Cold cells VOID (method bug, mine).** `gather_bench.py` ran the variants of a cell back to back in one process on
  the same ids; `POSIX_FADV_DONTNEED` cannot evict page-cache pages that are still mapped, and the first variant's
  gather had mapped exactly those pages, so every later variant ran warm (e.g. stock's own gather, the same algorithm
  as "old", came out 31 → 0.26 ms "cold"). Every cold old-vs-new and T2 nofill-vs-fill number here is withdrawn;
  the re-run uses one process per variant and its own ids.
- **T1, warm (valid):** prompt-size gathers **1.8–2.5× faster** (32,768 ids 17.1 → 9.5 ms; 131,072 ids 50.4 →
  20.3 ms; bytes equal in every cell) — but that is ~0.5 % of a 2,048-row chunk's ~2.8 s on EXL3, so **T1 cannot
  move warm TTFT**. Decode-size gathers got **slower**: 112 ids 0.116 → 0.195 ms, 448 ids 0.227 → 0.366 ms (the
  per-shard loop runs ~100 small fancy indexes over 128 shards where the old code did one over the file). Fix: one
  2-D row view per file (all shards sit in one file, contiguous), one gather per file → T1 v2.
- **T2, warm (valid):** adaptive within noise of no fill (112/448/896 ids: 0.200/0.355/0.419 vs 0.205/0.363/0.430 ms)
  — as hypothesised; forced fill on a warm table costs +0.45/+1.17/+1.92 ms (hypothesis +0.2…+2), so the adaptive
  gate is needed. Cold gain: void (above).
- **T4 closed: neither candidate.** Two stock 0.6.0 EXL3 starts with caches dropped, residency sampled every 2 s:
  the table goes 0 → 99.4 % within ~20 s of launch and stays at 100 % (no drop after layer 1), and the lock takes
  **0.6 s at 262144/8 and at 65536/4**. The 22.2 s in §5an was the first start after the 0.6.0 install, which built
  the CUDA extensions (nvcc) during the load; the likeliest reading is that the build's memory evicted the table, not
  a loader defect (not separately proven; the same shape no longer reproduces it).

- **Re-measure, one fresh process per cell** (`gather_bench2.py`, 5 seeds a cell, 150/150 bytes equal; T1 v2 =
  one row view per file). Medians, ms (0.6.0 → branch without the cold fill → branch with it):

  | ids | cold | warm |
  |---|---|---|
  | 112 (one round, c=1) | 23.8 → 7.36 → **0.80** | 0.112 → 0.041 → 0.022 |
  | 448 (c=4) | 95.9 → 29.1 → **2.27** | 0.255 → 0.087 → 0.037 |
  | 896 (c=8) | 184.1 → 58.0 → **3.99** | 0.403 → 0.137 → 0.058 |
  | 32,768 (a 2,048-token chunk) | 5,461 → 269 → 266 | 12.9 → 4.1 → 4.7 |
  | 131,072 (an 8,192-token chunk) | 18,584 → 1,112 → 1,116 | 51.4 → 8.8 → 8.9 |

  Against the hypothesis: **cold decode-size gathers 30–46× faster** with both changes (hypothesis 4–12×); T1 alone
  is already 3.2× faster cold (hypothesis ≈ old: its views carry `MADV_RANDOM`, so a fault reads one page instead
  of 0.6.0's default readahead window); **cold prompt chunks 17–20× faster** (hypothesis 5–15×); **warm**, T1 v2
  is now faster than 0.6.0 at every size (2.7–6× ; v1's decode-size regression is gone). The warm t1t2 < t1 at
  decode sizes is the extra warm-up gather t1t2 gets, not the fill.
- **What it buys a server:** warm, almost nothing (a round's gather is 0.1 ms of ~13 ms; a chunk's 13 ms of
  ~2.8 s). Cold, a lot: a decode round on a cold table drops from 24–184 ms to 1–4 ms of gather, and a cold
  2,048-token chunk's gather from 5.5 s to 0.27 s. A table is cold when memory pressure has reclaimed it: our
  checkpoint's table is not locked under 0.6.0 (§5an), and 0.6.0's KV growth takes memory from it by design. The
  server-level A/B therefore belongs in the long mix on our checkpoint (a big suite, at the end).

### 5ap. Where TensorFold's prompt time goes: 99 % GPU-busy, the experts' per-window work; chunk size is not the lever

Hypotheses `tools/tfprof/HYPOTHESIS.md`; data `data/tfprof/`, `data/tfsweep/`. One cold 8k prompt per checkpoint
under nsys (0.6.0 stock, one stream).

| | EXL3 3.05 bpw (11.75 s wall) | our `mtpfp4`, bf16 prompts (6.11 s wall) |
|---|---|---|
| GPU kernel time | 11.65 s (**99 %**) | 5.89 s (**96 %**) |
| routed experts | `tf_exl3x::grouped_kernel` **55.9 %** (882 calls), grouping 8.1 %, epilogues 7.3 %, unpack/rotate 3.7 % → **~75 %** | `nvfp4_expert_kernel` ×2 **36.7 %** (245 + 245 calls) |
| dense projections | `_gemm` 5.6 %, `_f16_mm` 4.7 % | `qmmf_kernel` (block FP8) **18.5 %**, `_b16mm` 7.2 %, `_fp4mm` 5.6 % |
| attention (`_chunks`) | 2.4 % | 4.9 % |
| DeltaNet chain | 1.6 % | 3.2 % |
| hyper-connections | 3.5 % | 9.5 % |

- **Hypothesis met on both counts that matter:** GPU-busy ≥ 85 % (99 / 96 %), experts ≥ 50 % on EXL3 (~75 %),
  NVFP4 experts ≥ 40 % on ours (36.7 %, just under; dense FP8 projections 18.5 % inside the 15–30 % expected).
  Busy is not efficient: at ~6.8 GFLOP a token these prefills reach ~5 (EXL3) and ~9 (ours) TFLOPS against vLLM's
  ~21 on our checkpoint — few rows per expert per call, EXL3's trellis decode per call, and W4A16 instead of
  vLLM's W4A4 (FP4 tensor cores).
- **Chunk-size sweep on EXL3: null, and the profile says why.** `TF_PREFILL_ROWS` 2048 / 4096 / 8192 (one start
  each, override confirmed: startup estimate 56.0 → 59.2 GiB): TTFT 8k **11.24 / 11.21 / 11.24 s**, 32k **45.71 /
  45.50 / 45.55 s**; greedy 64-token replies byte-identical in all three (`1a329bc1f8568edf`: chunk size does not
  change bits). The EXL3 routed-expert path runs in windows of **`MOE_WINDOW = 1024` rows** whatever the chunk
  (`exl3_pack.py:17`, "its grouping keeps every pick in 48 KB of shared memory"): 8,221 tokens → 9 windows × 2 calls ×
  49 layers ≈ the 882 grouped-kernel calls counted. The hypothesis (−20…−45 %) was built on the wrong knob. The real
  one is the window: 1,024 rows × 11 slots × 4 B = 45 KB; 2,048 would need 90 KB, inside GB10's opt-in maximum
  (~99 KB) once the attribute is raised — TF#151 does that, which the maintainer plans for 0.6.1. Our checkpoint's
  NVFP4 experts do run per 2,048-row chunk (245 calls = 5 chunks × 49), so the chunk sweep is still open there.
- **T9 v1 (shortest-first order): no effect** (small mix, 1 × 32k + 4 × 2k every 10 s, stock vs branch): short
  requests' first tokens at 46.5 / 39.3 / 32.1 / 22.4 s (stock) vs 51.8 / 39.0 / 26.1 / 22.1 s (branch), i.e. all
  of them wait for the 32k prompt to end (~45 s). Cause, from the code: with no stream decoding, `_fill` runs pass
  after pass until the filling prompts are done and never returns to the scheduler, so a new request is not even
  admitted until then; the order rule never sees two prompts. Exactness on the branch (levels 1, 2): alone 36/36,
  serial 12/12 equal. Fix (T9 v2): `_fill` also returns between passes when a
  foreground request waits (`MultiDecoder.arrived`, set by the scheduler to `waiting.foreground`).

### 5aq. Round 2: T9 v2 cuts short-request TTFT 22–47 s → 6–8 s; prompt cost is per row, so neither window nor chunk helps

Hypotheses `tools/tfprof/HYPOTHESIS.md` "Round 2"; data `data/tfsweep2/`. One start per arm, EXL3 unless noted.

- **T9 v2 works** (`ple-gather` `2e07da5`: a request that arrives while a lone prompt fills is admitted between its
  passes, then the shortest prompt fills first). Same small mix as §5ap (1 × 32k at t=0, 4 × 2k at 5/15/25/35 s):

  | | short 0 / 1 / 2 / 3 first token | the 32k prompt's first token |
  |---|---|---|
  | stock 0.6.0 (§5ap) | 46.5 / 39.3 / 32.1 / 22.4 s | 45.5 s |
  | branch, T9 v2 | **6.19 / 7.50 / 6.07 / 7.70 s** | 56.7 s (+11.2 s) |

  Inside the hypothesis (2.5–8 s; long prompt +≤ 12 s). Exactness at levels 1, 2: alone 36/36, serial 12/12.
- **EXL3 `MOE_WINDOW`** (branch + TF#151's smem attribute): TTFT 8k / 32k **11.72 / 47.94 s at 512, 11.27 /
  45.75 s at 1024, 11.09 / 44.71 s at 2048** → 512 +4.0 / +4.8 %, 2048 **−1.6 / −2.3 %**. Hypothesis (per-call cost:
  512 +40…60 %, 2048 −20…−30 %) **refuted**: the grouped expert kernel's time is per row, with only a few percent
  per call. Greedy reply byte-identical in all three (`1a329bc1f8568edf`).
- **Our checkpoint, `TF_PREFILL_ROWS`** (NVFP4 experts per chunk): 8k / 32k **6.05 / 24.26 s at 2048, 6.00 / 23.67 s
  at 4096, 6.09 / 24.06 s at 8192** → −0.8 / −2.4 % and +0.7 / −0.8 %: null (hypothesis −10…−25 % refuted). Greedy
  reply identical in all three (`551d0cf417be30bc`). So the L2/swizzle question for bigger chunks is moot too.
- **Reading:** on both checkpoints TensorFold's prompt time scales with rows at a fixed, low efficiency (~5 and ~9
  TFLOPS, §5ap); batching more rows per call or per chunk does not change it. What is left is how each kernel
  computes a row: W4A16/bf16 MMAs and the EXL3 trellis decode per row-tile, against vLLM's FP4 tensor cores (W4A4).
  That is the opt-in W4A4 question (its own issue), not a scheduling or chunking change.

### 5ar. Quant zoo on TensorFold: no silent garbage anywhere; NVFP4 dense layers crash late on stock, #176 refuses them up front

User: "we have a zoo of different quants for some components. we could try if all combinations are handled
properly". Every Flash-Next variant on disk, one start each, `--context 32768 --parallel 1`, three fixed greedy
prompts (64 tokens) and `/health` draft counters. Predictions `tools/tfzoo/HYPOTHESIS.md`; data `data/tfzoo/`.

| variant (what differs) | stock 0.6.0 | PR #176 branch |
|---|---|---|
| `fp8head` (block-FP8 dense + head, NVFP4 experts, bf16 MTP) | loads; replies sane; MTP 87/95 | — |
| `mtpfp4` (+ NVFP4 MTP experts) | loads; sane; 86/99 | loads; identical replies and counts |
| `mtpfp4-gdnbf16` (bf16 DeltaNet) | loads; sane; 87/94 | — |
| `mtpfp4-plebf16` (bf16 n-gram table) | loads; sane; 86/99 | — |
| `exl3` (EXL3 3.05 bpw) | loads; sane; 84/102 | — |
| `mtpfp8` (per-tensor FP8 MTP experts) | **refused at the check** (lists the accepted formats; does not name the FP8 layers) | — |
| `mtpfp4d` (NVFP4 MTP dense + fc) | **load error after ~1.5 min**: `torch.cat` shape mismatch (1280 vs 2560) in `b16_from_rows` | **refused at the check**: "NVFP4 in the routed experts only; … 20 other layer(s), e.g. model.mtp.fc_embedding" |
| `mtpfp4-gdn4` (W4A16 NVFP4 DeltaNet) | **load error**: same shape mismatch | **refused at the check**: "… 108 other layer(s), e.g. …layers.0.linear_attn.in_proj_qkv" |

- **Every prediction held** (the acceptance band was set too low: 82–92 % on these short greedy prompts, not 55–75 %).
  All five loading variants give the same three greedy replies (primes, Paris, a factorial function).
- **No variant loads and decodes garbage silently.** The packed NVFP4 dense weights do not survive as far as decoding:
  their halved K (`K/2` bytes a row) breaks a `torch.cat` while the stacks are built, so stock fails loudly, but late
  (after loading most weights) and with a message about tensor sizes, not formats.
- **PR #176's refusals are right, not harsher than needed:** stock does not serve `mtpfp4d` or `mtpfp4-gdn4` at all,
  so the branch only turns a late shape crash into an immediate refusal that names a layer. Worth a line on #176
  (draft only). `mtpfp8`'s refusal could name its offending layers the same way (a separate, minor topic).
- Not covered: combinations nobody built (e.g. MXFP8 dense + NVFP4 head); TensorFold's tiny synthetic checkpoint
  generator could cover every per-component format as a GPU unit test.
- **`mtpfp8` fixed on branch `pr-mtp-fp8`** (`e44d367`, user: "can we fix mtpfp8?"): FP8 MTP experts (per tensor, per
  row or 128×128 blocks) are dequantized to bf16 at load and re-quantized as TensorFold already does for bf16 drafter
  experts; FP8 in the main experts stays refused; the check accepts FP8 on MTP expert layers only. Real weights:
  loads, the same three greedy replies, MTP **86/93** (vs `mtpfp4` 86/99; hypothesis ±5 pp, +5.6 — three short
  prompts, read as "no worse"); `mtpfp4` on the branch identical to stock. Tiny-checkpoint test: FP8 experts draft and
  accept exactly as stacked bf16 experts holding the same dequantized values, and drafted == serial (greedy and
  sampled, depths 2/4, confidence 0/0.3). A first run of that test failed on my own bug (drafting without
  re-prefilling after the serial run), not the change. Covers NVIDIA's block-FP8 MTP layout in code; not measured
  (we hold no such checkpoint).

### 5as. The n-gram gather work (T1 + T2) does nothing for a server under load: no PR

User: "skip queued swt ... run the other tests", then "do only important tests": only the gather A/B ran (the
fill-order big mix and the full FP8-prompt sweep were dropped: #174 is accepted, and the maintainer declined FP8
prompts for Flash Next). Our checkpoint `mtpfp4`, `--context 262144 --parallel 8`, `longmix.py` (4 × ~120k prompts at
t=0, 8 × 2k every 15 s), one start per arm. Predictions `tools/tfbig/HYPOTHESIS.md`; data `data/tfbig/`.

| | stock 0.6.0 | `pr-ngram-gather` (T1 + T2) |
|---|---|---|
| errors / out-of-memory stops | 0 / 0 | 0 / 0 |
| MemAvailable minimum | 21.8 GiB | 20.9 GiB |
| major faults over the run | **1,114,573** | **1,339,940** |
| the four long prompts' first tokens | 98 / 197 / 294 / 392 s | 100 / 202 / 301 / 400 s |
| the short requests' first tokens | 297–389 s | 306–397 s |
| short requests' decode | 3.2–24.2 tok/s | 3.6–26.0 tok/s |

- **Stock against its prediction:** no stops, MemAvailable far above the reserve, short requests 297–389 s (predicted
  250–400 s), and the table faults heavily (1.1 M major faults, predicted ≥ 10k): on this checkpoint the 47.7 GiB
  FP8 table is not locked and does not stay resident beside the weights and growing caches.
- **T1 + T2: null to slightly worse** (~2 % later first tokens, +20 % major faults, one start each). T1 cannot act on
  this checkpoint (FP8 table, not EXL3); T2's fill touches whole pages for decode-sized gathers but the faults come from
  the prompt gathers (~7.7 M row lookups for 480k prompt tokens), which the pool already threads. The micro-benchmark
  gains (§5ao: cold decode gathers 30–46×) do not reach a server, so **no gather PR** — the branch stays local.
- The real cost here is the table not staying resident (1.1 M faults in 400 s). Locking it would need ~48 GiB that this
  checkpoint's shape does not leave; a smaller table (NVFP4, 28.6 GiB, as hibrid48) or the EXL3 5-bit one (30 GiB,
  locked in §5an) is the lever, not the gather.

### 5at. TensorFold's host-side cost: the draft chain is GPU-bound, the concurrent path is eager (~10 %), and a lone request on a multi-stream server loses its graphs (+11–13 %)

User: "go for it", after the agenda mapping (tensorfold-opportunities, "Speed-of-light agenda"). TF 0.6.0 (`c464617`),
EXL3 3.05bpw, greedy, 512 tokens of code per stream after a 64-token warm-up, one process per arm, two starts per
timing arm (no nsys), then one nsys run per arm. Probe `tools/tfhost/host_probe.py`, duty cycle `tools/tfhost/duty.py`
(union of kernel, memcpy, memset and graph-replay intervals over the capture window). Predictions
`tools/tfhost/HYPOTHESIS.md`; data `data/tfhost/`.

| arm | ms/token per stream, start a / b | aggregate tok/s | GPU busy (nsys) | idle per token | kernel launches per token |
|---|---|---|---|---|---|
| S1 one stream, graphs (default) | 12.094 / 12.190 | 82.7 / 82.0 | **97.6 %** | 0.29 ms | 25.6 + 1.44 graph replays |
| S0 one stream, eager | 13.554 / 13.532 | 73.8 / 73.9 | 88.3 % | 1.60 ms | 557 |
| M2 streams=2, c=2 (eager) | 17.877 / 17.933 | 111.9 / 111.5 | 87.8 % | 1.13 ms | 308 |
| M4 streams=4, c=4 (eager) | 25.249 / 25.327 | 158.4 / 157.9 | 89.5 % | 0.68 ms | 161 |
| L4 streams=4 engine, ONE request | **13.625 / 13.518** | 73.4 / 74.0 | — | — | — |

All arms write the same tokens for the shared prompt (`1bb116eb6ff5` in S1, S0, M2, M4 and L4; streams 1–3 identical
across M2/M4): the exactness contract holds. nsys adds 1.4 % (S0) to 3 % (M2) to eager wall time, so the traced idle
slightly overstates the untraced one.

- **Draft-chain host round trips (agenda item 2): closed.** With graphs the serial path is 97.6 % GPU-busy; all host
  time, the per-draft round trip included, is ≤ 0.29 ms per token (~1 ms per round). Removing it entirely is worth
  ≤ 2.4 %, not the restructure of `draft()` a device-side chain needs. Predicted 75–90 % busy; the out-of-range result
  says TF's one-stream graph path already hides its Python between replays.
- **Graphs on the serial path are worth 9.9–11.0 %** (S0 vs S1, pairwise per round), all of it idle removed: S0's GPU
  busy time (6.21 s) is S1's whole wall (6.30 s). Predicted 10–30 %: low end.
- **The concurrent path (agenda item 3) runs eager at 88–90 % busy**, the same idle share graphs removed on the serial
  path, so graphs there are worth **~9–10 % at c=2 and c=4** (upper bound from the traced idle). Predicted 40–75 %
  busy: out of range high, eager launch costs ~2.9 µs per kernel on the X925s, cheaper than assumed. Scaling over S1:
  M2 1.35–1.36×, M4 1.92× (both in range).
- **New, the concrete one: a lone request on a multi-stream engine runs eager.** L4 is S0's speed, +10.9…+12.7 % per
  token over S1 (pairwise). `qwen4_exp`'s `MultiDecoder` never uses the one-stream graphs (`multi.py:33`, "eager: no
  CUDA graphs"), while `qwen3_5_moe`'s does ("a lone stream replays `graphs`", `qwen3_5_moe/cuda/multi.py:44`,
  `:225-233`). A server started with `--parallel` > 1 therefore serves every single-user request ~11 % slower than
  `--parallel 1`. Porting the qwen3_5_moe pattern is the cheap fix; full graphs for c ≥ 2 are the larger, harder one.

Nothing posted. Upstream candidate (one topic): the lone-stream graph replay for `qwen4_exp`, needs the user's go.

### 5au. TensorFold 0.6.1's lone-stream graphs recapture on every slot resize (+35 % per token); fixed, branch `solo-graph-keep`

User: "can we fix graphs on single and multiple streams?" (after §5at). Field check first: the unreleased 0.6.1 port
(`origin/pr-141-0.6.1` = `cb5101d`, ashhart's port of #141) already adds lone-stream graphs (`multi_solo.py`: the
stream moves into a graph slot after its deferred DeltaNet rows are flushed, then replays the serial graphs). Measured
it with §5at's probe, worktree `~/git/tf-061`, two starts per arm. Predictions `tools/tfhost/HYPOTHESIS.md` rounds 2–3;
data `data/tf061/`, `data/tf061fix/`.

| arm | 0.6.0 | 0.6.1 port | 0.6.1 + fix |
|---|---|---|---|
| S1 one stream, ms/token | 12.09 / 12.19 | 12.33 / 12.15 | 12.18 / 12.19 |
| L4 streams=4, one request, ms/token | 13.63 / 13.52 (eager) | **18.41 / 18.56** | **12.18 / 12.20** |
| M2 two streams, tok/s | 111.9 / 111.5 | 99.8 / 99.1 | 111.7 / 110.8 |
| M4 four streams, tok/s | 158.4 / 157.9 | 146.8 / 145.1 | 157.6 / 157.8 |
| graph captures inside the measured reply (L4 / M2) | — | 23 (3.02 s) / 8 (0.93 s) | 0 / 0 |

Every reply byte-identical across all three builds (`1bb116eb6ff5`, `fc0e045ffadf`, …).

- **Cause (measured, `CAPTURE_LOG=1`):** `_grow` and `_shrink` call `_state_changed`, which replaces the solo slot's
  `Graphs`. A slot starts at 256 rows, grows to 8,192 once the context passes it, and shrinks back after the request,
  so every lone request recaptures ~23 graphs at ~131 ms each mid-decode. Counterfactual: with the slot grown during
  the warm-up (`WARM_TOKENS=600`) the same request takes 0 captures and runs at 12.13 ms/token (= S1). M2/M4 regress
  the same way: stream 0 starts alone, enters the slot, and the slot resizes.
- **Predicted** L4 = S1 ±2 % on the port: out of range (+49 %), which is what sent us to the capture count.
- **Fix** (`jschmied/TensorFold` branch `solo-graph-keep`, `153817c`, on `cb5101d`, `multi.py` +15/−5): the solo
  slot keeps its rows when idle and gives them back only under memory pressure (evicted kept end, newest stream ended,
  or an idle solo slot freed for another stream's growth); it grows by doubling (≥ 8,192), so a long session
  recaptures a handful of times; `warm()` captures its graphs last, after the warm request has grown the slot. All
  arms in the round-3 range: 0 captures, L4 = S1, M2/M4 back to 0.6.0.
- **Host tests:** 14 multi-decoder files, baseline and fix fail the identical 28 (with a pytest plugin giving the
  port's new attributes class defaults — the port's own tests build `MultiDecoder` without `__init__` and hit
  `self.solo`; without the plugin both fail the identical 32). All 8 growing-cache and slot tests pass on both.
- **Not measured:** int8/int4 KV, the two-rank plan path (`_is_solo` sees through a plan's stand-in, the new
  eviction branch is skipped while planning), memory pressure with the slot held at its rows.
- **c ≥ 2 graphs** (the other ~9–10 %, §5at) are not in 0.6.1 either. Scoped from source: the multi-stream round's
  attention already reads cache pointers from a device table (`attn_multi.Step`), DeltaNet the same
  (`gdn_multi.Tables`); blockers are the per-round tables allocated fresh (need persistent buffers), the per-stream
  sparse-select launch with direct pointers (`attn_multi.layer`, needs a table-driven kernel), and launch dims tied to
  the round's shape (key on total rows, streams, parity, context bucket). Graphs built that way would also survive
  resizes, so they would replace the solo slot's copy-in. Not started.

Posted 2026-10-01 (user's go): PR #180 (`6a3a8b1`, with a host test) against `pr-141-0.6.1`, and a short comment on #141.

### 5av. T7: our 32k draft vocabulary on TensorFold — cheaper head, worse prose acceptance: no proposal

TF 0.6.0, EXL3, greedy, 512 tokens per stream, default draft list (79,591 ids: every id below 65,536 + stdlib-code
frequency, `docs/recipes/qwen3.8-flash-next.md`) vs ours (`tools/draft_vocab/draft_vocab_32768.txt`, det-135), S1 and
M4, alternating, two rounds; code prompts, then `PROMPT_SET=prose` (English, German, Chinese, French). Hypothesis
`tools/tfhost/HYPOTHESIS.md` rounds 4–5; data `data/tfdv/`, `data/tfdv2/`. Hashes identical in every arm.

| | default | 32k |
|---|---|---|
| code, one stream, ms/token | 12.13 / 12.22 | 11.13 / 11.15 |
| code, one stream, ms per round | 42.2 / 42.5 | 41.3 / 41.3 (−2.2…−2.7 %) |
| code, rounds per prompt (4 prompts) | 145 / 140 / 136 / 115 = 536 | 136 / 140 / 136 / 120 = 532 |
| prose, one stream (English), ms/token | 18.65 / 18.66 | **19.79 / 19.77 (+6.1 %)** |
| prose, rounds (en / de / zh / fr) | 263 / 229 / 415 / 288 = 1,195 | 283 / 301 / 373 / 323 = **1,280 (+7.1 %)** |

- **The head is cheaper by 1.4–2.7 % a round**; everything else is acceptance, and acceptance is prompt-specific: the
  single code prompt of S1 gained 9 rounds (hence its −8 %), German prose lost 31 %, French 12 %, Chinese gained 10 %.
- **Coverage does not predict it:** on 443k stdlib-code tokens the default covers 99.99 %, ours 97.48 %; our list holds
  1,826 ids the default lacks, and drops most ids below 65,536.
- **M4 aggregate tok/s is set by the slowest stream** (all four must finish): its +2.8 % (prose) and +4 % (code) follow
  the slowest prompt's rounds, not the head. Read c=4 per-round, not aggregate, in future draft-vocab A/Bs.
- **Verdict:** T7 closed, no proposal — TF's default list is the better general choice (the kill rule in the
  hypothesis: total prose rounds +7.1 % > 5 %). The vLLM 32k slice (det-135) was measured against the full 248k head,
  not against a 79k list that keeps all low ids.

### 5aw. TensorFold's draft depth 6 / confidence 0.7 sits at the optimum (±1 %): depth/τ is not a lever there (#136)

User: "check #136, i think we measured a lot here" (#136: choose the draft count by expected committed tokens per ms;
ashhart: an A/B by net tok/s on held-out prompts before any default changes). TF 0.6.0 `--mtp-drafts` /
`--mtp-confidence` through the probe (`MTP_DEPTH`, `MTP_CONF`), EXL3, greedy, 512 tokens, code and prose sets
(English/German/Chinese/French), S1 and M4, two rounds. Hypothesis `tools/tfhost/HYPOTHESIS.md` round 6; data
`data/tfdepth/`. Hashes identical in every cell.

| S1 ms/token | 6 / 0.7 (default) | 6 / 0.8 | 7 / 0.8 | 8 / 0.8 |
|---|---|---|---|---|
| code | 12.16 / 12.22 | 12.27 / 12.27 | 12.06 / 12.08 | 11.99 / 12.07 |
| prose (en) | 18.63 / 18.63 | 18.70 / 18.77 | 18.74 / 18.72 | 18.75 / 18.76 |
| code rounds / drafted / accepted | 145 / 502 / 367 | 155 / 444 / 356 | 148 / 465 / 363 | 147 / 465 / 364 |

M4 (sum of the four streams' decode seconds, not aggregate tok/s, §5av): code 48.3 / 48.6 / 48.6 / 48.7 s, prose
78.8 / 78.2 / 78.6 / 79.6 s — every cell within ±1 % of the default.

- **All inside the hypothesis.** 7/0.8 vs default: code −0.9 % (predicted 0…−3 %), prose +0.5 % (−2…+1 %); 8/0.8
  ≈ 7/0.8; 6/0.8 +0.6…+0.9 % slower.
- **Consistent with our vLLM replay once the baselines match:** §5ah's +10 % code / +12.8 % prose is depth 7 + stop
  against a *fixed* depth 5; against a depth-6/τ-0.7 stop (TF's default) the same replay predicts +2 % code / +0.2 %
  prose, measured +0.9 % / −0.5 %. The gain is the in-round stop itself, which TF already has.
- **Deeper chains barely draft deeper at τ 0.8:** the chain rarely survives to position 6 (drafted 465 vs 502 at
  depth 6/0.7), so depth 7/8 add almost nothing.
- **For #136:** on these prompts a perfect per-round depth choice has ≈ 1 % left over TF's default; a depth rule from
  *recent* acceptance lost in our replay (§5v: code −0.8…−8.2 %). Posted 2026-10-01: https://github.com/ashhart/TensorFold/issues/136#issuecomment-5931347120

### 5ax. T13: TensorFold's EXL3 routed experts launched a program per 16-row tile of the whole window; a tile list cuts prefill 9.4–10.2 %, bit-identical

User: "write this as next todo", "start with it". The plan was decode-once reuse; the measurements moved it. TF 0.6.0
`cuda/exl3/experts*.{cu,cuh,py}`, Flash Next shapes (D 2560, expert width 640, 512 routed + shared = 11 slots, K2 6,
`mul1`), prefill windows of 1,024 rows. Hypotheses and stop rules `tools/tfexl3/HYPOTHESIS.md` (rounds 1–6); data
`data/tfexl3/`; branch `exl3-prefill` (worktree `~/git/tf-exl3-prefill`).

1. **Decode once into fp16 (option 2) is dead:** dequantising a window's experts and a padded bmm took 59 ms against
   the grouped kernel's 21 ms: fp16 weights are ~5 GB a window against ~0.95 GB of 3-bit trellis. In-register decode is right.
2. **ncu** (`grouped_ncu.py`): memory throughput 12–14 %, SM 17–19 %, 167 registers → 3 blocks/SM (25 % occupancy),
   ~16 cycles per issued instruction, tensor instructions 2–3 % of all: latency-bound. **Capping registers**
   (launch bounds 4/5 blocks): null / +12 % (spills) — rejected.
3. **The grid:** z = mats·splits·⌈maxm/16⌉ with maxm = the window, so every expert got 64 row-tile programs although a
   routed expert has ~20 rows; the shared expert (all 1,024 rows) made trimming maxm impossible on the real path. Up to
   97 % of 1.3 M gate/up programs started, read `members`, and returned.
4. **Fix (tile list):** `group_kernel` also writes `tiles[]` = (place << 8 | member tile) for every non-empty tile and
   `tcount`; `grouped_kernel` takes its expert and member tile from the list; grid (⌈R·slots/16⌉ + maxu, N blocks,
   mats·splits). Each program's arithmetic is unchanged.

| | stock | tile list | Δ |
|---|---|---|---|
| `routed()` R 1,024 (3 rounds), ms | 26.42 / 26.39 / 26.41 | 18.69 / 18.67 / 18.70 | **−29 %** |
| `routed()` R 64 / 8 / 1, ms | 3.87–3.92 / 0.74–0.78 / 0.077–0.081 | 3.55–3.60 / 0.75–0.78 / 0.078–0.081 | −8 % / ±0 / ±0 |
| EXL3 prefill 8k, s (2 rounds, real checkpoint) | 11.42 / 11.47 | **10.35 / 10.32** | −9.4…−10.0 % |
| EXL3 prefill 32k, s | 46.18 / 46.38 | **41.79 / 41.66** | −9.5…−10.2 % |

- **Bit-identical:** `routed()` output hashes equal stock at R 1/8/64/1,024; the first 16 generated tokens after the 8k
  and 32k prompts equal stock in every arm; TF's EXL3 GPU tests pass (incl. GLM bit-identity and graph replay) and a new
  test checks the tile list is exactly the non-empty tiles.
- **Against the hypothesis:** kernel −29 % beat the predicted −13…−19 %; end to end −9.4…−10.2 % sits at the bottom of
  −10…−18 % (the grouped launches are ~56 % of an 8k prefill; the rest of `routed()` — the single-block `group_kernel`,
  rotation, epilogues — and the dense layers are unchanged).
- **Next, still in the kernel:** decode reuse across row tiles needs more rows per expert per call (bigger windows:
  `group_kernel`'s shared memory caps them at 1,024 rows × 11 slots, #151); `group_kernel` is one block scanning all
  picks per expert (not yet measured).
- Upstream: **PR #184** (2026-10-01, base `pr-141-0.6.1`, `90fafa8`; 0.6.1 rebase adds #151's static-smem count and
  a 16-bit tile field): https://github.com/ashhart/TensorFold/pull/184

**§5au addendum (2026-10-01, BHCC2025's two-rank run on #141):** #180's `_is_solo` compared rank 0's plan `Shadow`
with the real slot and never matched while planning, so on two ranks #180 changed only the warm-up order — their
`cb5101d` vs `+ #180` spread (−5…+2 %) is run-to-run variation, and their lone-request gap (~11 % on `--parallel 8`)
is the unfixed recapture. Fix `6ada832` (compare the slots behind the Shadows; idle-slot release allowed while planning,
recorded for rank 1) + a Shadow test that fails on `6a3a8b1`; pushed to #180, reply posted. Two ranks not testable here.

**§5au addendum 2 (2026-10-01, user: "#180 need a fix more?") — yes: two more paths.** `tools/tfhost/solo_switch.py`
(a `--parallel 4` engine, requests A, B, C with three prompts, then A2 extending A; 256 tokens each, greedy; data
`data/tf061fix/switch-*.txt`):

| captures, ms/token | 0.6.1 port | #180 `6ada832` | + `5824347` |
|---|---|---|---|
| A | 33, 31.6 | 0, 14.0 | 0, 14.0 |
| B (new prompt) | 32, 29.4 | **32, 29.3** | 0, 12.7 |
| C (new prompt) | 30, 27.3 | **30, 27.5** | 0, 11.7 |
| A2 (resumes A) | 19, 23.0 | **19, 22.1** | 0, 12.1 |

- **Path 1 (prefix cache):** a finished lone request keeps its prompt end in the graph slot; the next lone request
  with another prompt lands elsewhere and `_move_to_solo` made *that* slot the graph slot (`_state_changed`). Fix:
  move the kept end to a free slot of the same rows (`copy_from`; snapshot unchanged), keep the graph slot. No free
  slot that fits without evicting → the old swap; two ranks keep the swap (kept ends are keyed by slot in the plan).
- **Path 2 (latent crash in #180):** with the graph slot keeping its rows, a stream from a smaller slot no longer has
  the same geometry and `copy_from` failed on tensor sizes. Fix: `copy_from` takes a smaller source into the first rows.
- Every reply byte-identical to the port's (hashes), incl. A2 resumed from the moved prompt end. Host tests: 211 passed,
  same 29 unrelated failures; the two new tests fail on `6ada832`; `tests/cuda/test_flashnext_multi.py` 27/27 on both.
- Window lever (T13 round 8): MOE_WINDOW 2048 on #184: 8k 10.40/10.30 → 9.90/9.92 s, 32k 42.29/41.57 → 40.26/39.86 s
  (−3.2…−5.7 %), hashes identical; branch `exl3-window-2048`, not posted. routed() breakdown on #184: grouped 72 %,
  group_kernel 13 %, epilogues 11 %, rot_in 3 % (`data/tfexl3/prof184.txt`); parallel grouping in progress.
- **Lever 2, parallel grouping** (branch `exl3-group-parallel` on #184, `data/tfexl3/grp-*`): count (atomics) → one
  block's scan over the experts → a warp an expert compacting its picks with ballots (row order kept). Grouping
  2.53 → 0.083 ms a 1,024-row window (5.07 → 0.16 at 2,048), routed() 18.71/18.88 → 16.35/16.38 ms (−13 %), hashes
  identical at 1/8/64/1,024 rows, EXL3 GPU tests 76 passed. No picks in shared memory any more, so the grouping has no
  window limit (#151's opt-in launch test becomes "a window past the old ceiling groups exactly"). Not posted.
- **#184 review (2026-10-01, user-pasted; P3 + a reuse test):** grid.x = min(tile capacity, experts · ⌈maxm/16⌉) — short
  windows launch what they did before #184; new GPU test reuses one Scratch through 1,024 → 1 → 17 rows → no valid
  picks → 17 rows (77 EXL3 tests pass). Small-window cost vs `cb5101d` (`small_bench.py`, median of 5 × 2,000 calls,
  3 processes each): 1 row 76.6–78.3 vs 78.7–83.8 µs (noisy), 2–8 rows +0…+2 %, 16 rows ~+0.5 %; the rest is the
  second scan/tile writes in `group` and one more dependent load per program (`tcount` → `tiles`). Est. ≈ +0.5 % per
  decode round, below e2e noise (not measured e2e). Commit `0b851ce` on `exl3-tile-list-061`, not pushed.
- **Lever 2 end to end (rebased on #184 head `0b851ce`, `f71d25f`):** 8k prefill 10.27/10.30 → 9.41/9.39 s, 32k
  41.58/41.49 → 38.64/37.86 s (−6.9…−8.8 %), tokens identical; small windows: 1 row +0…+5 % (three launches + memset
  instead of one block), 2/4 rows even, 8/16 rows −1.3/−2.4 %. **PR #191** (stacked on #184). Cumulative on 0.6.1 at
  8k: 11.42 (stock) → 10.27 (#184) → 9.40 s (#191) = −18 %; window 2048 (−4…−5 %) not yet posted; lever 3 next.
- **#178 receipt (2026-10-01, ashhart asked which export):** `mtpfp8` = lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8 @
  `a9786bf` (205/206 weight shards byte-identical by HF lfs.oid; MTP-expert shard `model-bf16-00011` identical; one BF16
  shard ours) + two FP8 `quantized_layers` entries (both spellings) that lovedheart's config does not have. As
  published (`qwen38-flash-next-mtpfp8-asrel`, hardlinks + their algo set): stock 0.6.0 passes the check and fails at
  load (`KeyError … weight_scale_2`); #178 loads, same replies, 86/93 accepted. First asrel run void (kept the
  `model.mtp.*` entries). Posted: https://github.com/ashhart/TensorFold/pull/178#issuecomment-5934208411
- **#191 review (user-pasted, P2):** with the grouping's shared-memory ceiling gone, `grouped_kernel`'s
  `members[u * maxm + m]` overflowed int32 for windows the limits now accept (2,049 experts × 1,048,576 rows at
  u = 2,048). Fixed with a size_t offset (`5727ea6`, **PR #193**); audited the other
  offsets on the path (rotation, epilogues, combine, activation rows): already 64-bit. 77 EXL3 GPU tests pass, hashes
  unchanged. A test at the boundary would need ~8.6 GB of members — not added.
- **Lever 3 (two member tiles a program) — closed, slower:** bit-identical, but grouped launches +16 % at 1,024 rows
  and +64 % at 2,048 (`data/tfexl3/ms-*`). ncu: instructions −26 %, cycles per issued instruction 14.3 → 23.0 (NT 4
  doubles activation loads per mma and the programs; occupancy 25 % either way; NT 8 × MS 2 needs 64 KB reduction
  memory and ~230 registers). The grouped kernel is latency-bound; decode reuse needs an ILP redesign first (pipelined
  activation loads / cp.async). Branch `exl3-subtiles`, local only.
- **8k profile on the current stack** (#184 + #191 + #193, `data/tfexl3/p8k*`): grouped 54.5 %, dense prompt GEMMs
  (`_gemm` + `_f16_mm`) 13.1 %, epilogues 9.2 % (at DRAM bandwidth), `bfloat16_copy_kernel` 3.0 %, rot_in 2.7 %, HC glue
  6.2 %, attention 3.5 %, DeltaNet chain 2.0 %.
- **Lever 4b — the down epilogue writes the prompt's bf16 rows** (branch `exl3-bf16-rows` on #193, local): no fp32
  rows, no per-window copy; same fp32 value rounded to nearest even. 8k 9.62/9.61 → 9.19/9.09 s, 32k 38.77/38.77 →
  37.20/36.72 s (−4.1…−5.5 %), tokens identical, 78 EXL3 GPU tests (incl. bf16-out == fp32.to(bf16)). Not posted.
  Cumulative 8k on 0.6.1: 11.42 → 10.27 (#184) → 9.40 (#191) → **9.14 s** (−20 %); window 2048 not stacked yet.
