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
