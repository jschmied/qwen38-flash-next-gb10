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
  hashes (8/8). RecoverSSM-align's outputs are identical to RecoverSSM phase 1's, so align mode changes nothing.
  Divergence vs base-align is unchanged: median 26 tokens. Base-align also equals phase-1 base, byte for byte.
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
