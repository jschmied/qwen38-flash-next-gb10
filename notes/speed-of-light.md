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
