# Prefill plan — make TTFT better (2026-09-03)

Agent work on this box is TTFT-bound (`agentic-speed-is-ttft-bound`, finding 61: MTP does not
pay at 30-token turns). Baseline, served config (prefix cache on, batch 4096/8192, det MoE):

| prompt | TTFT | rate |
| --- | --- | --- |
| 7,503 tok | 3.12 s | ~2,400 tok/s |
| 29,263 tok | 11.25 s | ~2,600 tok/s |

Goal: a measured TTFT reduction at both points with unchanged output (bit-identical under the
deterministic stack, or Δtop-1 ≤ noise vs the reference), each lever A/B'd with 3 starts.

## 0. Owed first (queue is busy until they land)

`mtpgrid0` → correction numbers for #53142/#54076/#53798; `kdetab` → e2e decode/TTFT numbers for
PR #55122; `kpytest` → the PR's test file against `_C_det`. Post after the user's go.

## 1. Profile before touching anything (½ day, box free)

`FN_PROF_DIR` is wired (torch profiler via `--profiler-config`). One start, `/start_profile` →
one 8k request → `/stop_profile`, same for 30k. Table: kernel time by component — GDN chunked
scan (triton, `VLLM_GDN_DECODE_KERNEL` also switches the prefill impl, finding 23), QSA indexer
logits + `persistent_topk`, QSA sparse attention, MoE (FlashInfer CUTLASS, M in the thousands),
dense/attention GEMMs, **PLE gather** (CPU, FP8 table RAM-resident ~51 GB, `VLLM_PLE_CPU_OFFLOAD=1`),
hyperconnections, and idle/CPU gaps. Decision gate: the top three consumers set the order below.
Do not skip this — the last profile was prefill-contaminated (`the-prefill-decode-confound.md`).

## 2. Levers, ordered by expected value ÷ cost

| # | lever | why | cost | risk |
| --- | --- | --- | --- | --- |
| A | **main build** (nightly aarch64 wheel `0.28.1rc1.dev345+g4cc0cb6f7` exists; arch `Qwen4ExpForConditionalGeneration` = our checkpoint) with #54513 (prefill/decode-split QSA indexer, up to 1.3× on the indexer, workspace 128→512 MiB) | biggest merged upstream prefill change since the preview | pip venv clone + the known reinstall casualties (FP8 shim, PLE backport `4e8b849b8d97`, det patches) | preview-only features may be missing; re-validate load defects (`flashnext-fp8mix-checkpoint`) |
| B | **PLE on GPU** instead of CPU gather — HashK-PLE (Death-By-Tokens, 51 → 12.8 GB, trainless) or a plain GPU-resident slice if memory allows | if the profile shows the CPU gather in the prefill critical path, this is the only lever that removes it | download (ask first) + loader work | quality: verify Δtop-1 vs reference; memory headroom at 128 GB |
| C | GDN prefill implementation A/B (`VLLM_GDN_DECODE_KERNEL` triton vs cuda — it switches prefill too) | two implementations exist; never timed for prefill | 2 arms × 3 starts, 40 min | cuda arm hangs at c~32 with FP8 GDN (serve-fnext comment) — test c=1 only |
| D | `--max-num-batched-tokens` on the winner of A/C (4096 / 8192 / 16384) | already measured on the preview: 16384 −7 % at 32k, worse at 4k; re-check on the new stack | 3 arms × 3 starts | none |
| E | MoE prefill backend at large M (`flashinfer_cutlass` vs `marlin`; b12x/cutedsl vetoed on this checkpoint) + autotune-cache warm vs cold | prefill M is thousands; the decode choice may not be the prefill choice | 2 arms | determinism: only `cutlass` has the det finalize |
| F | Kernel-det vs exact top-k (already in `kdetab`) | −4 pp TTFT vs the Python workaround | queued | — |
| G | Prefix-cache block granularity / align cost at 8k (finding: spec decode costs one prefix block) | second-order | later | — |

Not levers (measured or excluded): `cooperative_topk` (needs TMA, excluded on sm_12x); llama.cpp
#28136 (cold-start PLE read from SSD — our table is RAM-resident); MTP depth for TTFT.

## 2b. Field calibration (sweep 2026-09-03, `the-field.md`)

Every GB10 stack — vLLM or SGLang, PLE on CPU / NVMe / swap / **GPU (HashK)** — reports 2.3–2.7k
tok/s cold prefill; an RTX Pro 6000 does 11–13k. So prefill here is compute-bound near the hardware
line and lever **B (PLE on GPU) is not a prefill lever** (airawatraj: 2.4–2.5k with GPU-resident
HashK) — keep it only for memory. Realistic win from A/C/D/E: 5–15 %, unless the profile finds an
idle gap. The multiplier that is real is the warm prefix cache (~10× on repeated prefixes; we
already hold −40 % TTFT after turn 1 and one lost block under spec decode) — so lever G moves up:
maximise hit rate in the agent loop (block size, spec-decode block cost, what the client resends).
Also seen: crimsonjoo's vLLM recipe pays −17…−40 % prefill for its exact top-k default; PR #55122 is
the answer for them (comment after the user's go).

## 3. Method

- One runner per lever, `TTFT=1` arms (ttft.py, 8k + 30k, 3 requests), 3 starts, interleaved.
- Activation line per arm in the log; never two variables in one `Environment=`.
- Quality gate per winner: the deterministic stack makes outputs bit-comparable — compare the
  8k/30k completions against the current stack before adopting anything.
- Record every result as a numbered finding in `determinism-investigation.md`'s successor file
  `prefill-investigation.md`; raw data to `notes/data/`.

## 4. Order of execution

owed measurements → profile (1) → A (main build, because C/D/E should be measured on the stack we
will keep) → C, D, E on it → B if the profile says PLE gather matters → write-up + prod switch
(user decision).

## 5. Ideas that are ours, not the field's (2026-09-03)

Replicating the field buys ≤ 15 %. These are the directions nobody in the sweep is on, ranked by
expected TTFT effect in the agent loop ÷ effort:

1. **Spec decode must not cost a prefix block — CORRECTED: known upstream, stuck on scope.** The
   drop is deliberate code (`drop_eagle_block` in `get_computed_blocks`: EAGLE-family drafters need
   the hidden state of the last token, so the last full block is never served from cache). Upstream
   issue #53670 (Suppressor72, 08-25; our GB10 numbers are comment 2; Gemma-4 reproduction too) and
   the candidate fix PR #50897 (ZJY0516, lookahead-aware prefix hashing, +2434/−205 over 50 files,
   incl. mamba-align tests) — which we measured working on the 27B on 08-24 — is `needs-rebase`/DIRTY
   and reviewer ivanium asks for it to be split (PP deferred-finalize fix, KV-event revamp, the
   rest). What is ours here: (a) **measure it on Flash-Next** (hybrid + align + in-checkpoint MTP —
   nobody has; #53670 names the GDN layout explicitly), (b) if it holds, offer the split-out core
   ("the rest") as a small PR with the hybrid measurement, which is what unblocks it. Not novel
   as an idea; novel as the missing evidence and the missing small PR.
   **UPDATE 2026-09-03 (finding 80):** upstream did not wait for #50897 — #53388 (merged 09-01, in
   our main venv `0.28.1rc1.dev352`) adds the opt-in `disable_eagle_block_drop` on
   `SpeculativeConfig`, honoured for every `use_eagle()` method **including `mtp`**: the trailing
   block is kept instead of dropped. The preview build lacks it. So the experiment is now a
   one-flag A/B on the main build, no patch: `--speculative-config '{"method":"mtp",
   "num_speculative_tokens":3,"disable_eagle_block_drop":true}'`, prefix cache on, `agentloop2`
   turns, compare `prefix_cache_hits_total` deltas + TTFT per turn + MTP acceptance (the PR warns
   the proposed drafts change, so acceptance may move) against the same config with the flag off.
   Expected: hit rate back from ~43 % to ~69 % and the extra cold turn gone (finding in
   `spec-decode-prefix-cost-agentloop`). Needs the box (main-build serve, ~2×20 min) — after the
   night chain and `race50729`, on the user's go. **QUEUED 2026-09-04 09:10 as runner `blockdrop`** (user go
   "continue"): main venv, MTP n=3, prefix caching on, three interleaved starts of off/on via the new
   `FN_SPEC_NODROP=1` knob in `serve-fnmain.sh`; runs after `chunkredo` and `qsadump2`.
   **DONE 2026-09-04 12:40 — finding 94: warm turns −26 %, cached tokens +33 %, acceptance unchanged. CLOSED.**
   Also on main and not in the preview: #52789 "internal prefill checkpoints" (one forward pass
   for the whole prefill, Mamba checkpoint saved mid-kernel instead of splitting the forward at the
   last block boundary; 9–25 % TTFT claimed). The generic side is in (`num_prefill_checkpoint_blocks`
   in `kv_cache_interface.py`, `_needs_internal_checkpoint` in the cache manager) but only Kimi-K3's
   KDA layer implements the kernel side; the GDN backend does not claim it. For us it is a
   contribution candidate (GDN `chunk_gated_delta_rule` already checkpoints per chunk internally),
   not a free win.
2. **PLE gather off the critical path.** The n-gram lookup depends only on token ids, so every row
   for a whole prompt is known at tokenization time; on a cache hit the prefix's rows are not needed
   at all. If the profile shows the CPU gather inside prefill time, prefetch the next chunk's rows
   during the current chunk (one-chunk pipeline) and skip rows for cached blocks. Cheap, and the
   worker already streams H2D asynchronously (`ple_offload/worker.py`), so the plumbing exists.
3. **Index sharing across query positions in prefill.** *(2026-09-04: measured GO — findings 92 + 96: gather saved 90–98 %, loop 3× faster at M=64; design fixed at 4 rows/program, BN 64, 1 stage by the 99 KiB smem; ~−8/−9 % TTFT; awaiting the go for 2–3 days of kernel work.)* The QSA indexer scores every query token
   against every block (O(T × blocks); at 30k that is the bulk of the indexer cost). vLLM already
   shares one selection across MTP draft positions (`index_share_for_mtp_iteration`); for prefill,
   compute the selection every s-th query position and reuse it for neighbours. A quality trade —
   and we are the only stack that can measure it exactly (bit-comparable outputs, logprob
   divergence per layer). Gate: profile share of the indexer ≥ 20 %.
4. **Persist the hybrid state across restarts.** With bit-exact determinism, GDN state + QSA KV of a
   prefix is a pure function of the tokens: snapshot at align boundaries to host memory/NVMe and
   reload — a KV-connector for hybrid models. Big effort; the payoff is cold-start TTFT of long
   system prompts after every restart/eviction, which every recipe in the sweep pays.

Not novel, dropped: PLE placement, batch sizes, MoE backend sweeps, GDN kernel A/Bs (keep as
5–15 % housekeeping on the main build).

5. **Hyper-connection GEMMs at prefill M (re-opened 2026-09-03).** Per layer: grouped RMSNorm →
   down GEMM `[M,10240]→[M,320+4]` → silu → up GEMM `[M,320]→[M,10240]` → gate-mix (norm and mix are
   already fused Triton ops; the GEMMs are plain BF16 `ReplicatedLinear` → cuBLAS →
   `cutlass_80_wmma_tensorop_bf16`, an sm_80 path). FLOP share ≈ 5 % of the model, time share ≈ 25 %
   in the earlier prefill-heavy trace — i.e. the kernel runs far below the tensor-core rate for
   these skinny-K/skinny-N shapes on sm_121. At **decode** (M ≤ 8) three interventions were null
   because the shapes are latency-bound (`the-prefill-decode-confound.md`); at **prefill** (M in the
   thousands) that argument does not hold, so the same three arms (per-channel FP8 `scaled_mm`,
   blockwise FP8, a Triton/CUTLASS GEMM tuned for K=320) are un-measured where they can matter.
   Fusion (up GEMM + gate-mix epilogue, down GEMM + silu) is second-order: the gate round trip is
   ~16 GB per 8k prefill ≈ 60 ms ≈ 2 % of TTFT. Order: profile share (queued) → offline microbench
   of the two shapes at M ∈ {4096, 8192, 32768}: torch BF16 vs `scaled_mm` FP8 vs a Triton GEMM
   (minutes, idle GPU) → if ≥ 2× on the kernel, wire it through the existing `quant_method` dispatch
   (main already has `low_latency_gemm.py` for the decode side) and A/B TTFT. Nobody in the field
   touches the hyper-connections.

## 6. Night queue 2026-09-03 (user: "thats fine")

`night` unit: moeab (MoE backend prefill A/B, main) → mtpgrid0c (grid remainder) → kdetab (PR #55122 e2e)
→ chunke2e (Python-equivalent M-chunking on main at batch 8192, on/off) → `qsadump` (QSA block-selection
overlap for one 8k + one 30k prefill: consecutive-row Jaccard and union-per-tile vs per-row gather — the
go/no-go number for the tile-union sparse-attention kernel, §5 item 3 / kernel list #2). Deferred to a
GPU-free slot because they need compiles: deterministic top-k pass-cost breakdown (kernel list #4) and the
CUTLASS raster-swizzle experiment (#5). Results in `/opt/llm/runners/results/`, findings in the morning.

### Queue 2026-09-04 (user: "queue micro benchmarks for this" — the small-L2 question)

Chained on the box after `chunkredo` → `qsadump2` → `blockdrop`:

- **`moel2`** — `tools/moe_l2sweep.py`: NVFP4 grouped MoE GEMM vs M with balanced and random routing.
  Hypothesis: rows per expert = M·10/512; past one 128-row M-tile (M ≈ 6.5k) each expert's ~2.5 MB
  weight is streamed a second time — that is finding 78's "2× above the floor" and why batch 4096
  beat 8192. A step at the tile boundary confirms; then the lever is a ≤ 6k MoE chunk and, upstream,
  per-expert M-chunking in the grouped kernel (#55180's shape of fix).
- **`hcbench2`** — `tools/hc_kernels_bench.py`: the two hyper-connection Triton kernels
  (`_hc_combine_norm` 2.88 ms/call, `_hc_gate_mix` 1.55 ms/call at 4096 rows; 9.3 % of the 7.5k prefill
  in `prefprof`, 12.7 % at 30k, ~15 % of the main build's 8k TTFT now that the FP8 GEMM is fixed) against
  a torch bandwidth floor and re-tiled variants (all HC streams per program, block read once; rows×cols
  tiles for the mix). Stock runs at ~90–115 GB/s on a 273 GB/s part. Expected win if the variants
  reach the floor: ~1.6 + 0.8 ms per call → ~8 % of 8k TTFT, more at 30k. Numerics checked vs stock.
- **TODO later — 27B swizzle optimizations (user 2026-09-04 16:2x):** apply the finding-100 lever to the two
  GEMM paths the 27B actually runs, both larger than the L2 by 2.5–7×: (a) the per-channel FP8 W8A8 dispatch
  `scaled_mm_sm120_fp8_dispatch.cuh` (attention q/k/v/o, GDN qkv/z/out at 60–120 MiB, the eight FP8 MLP layers
  at 170 MiB) — same `cutlass_gemm_caller` plumbing, one scheduler argument, unmeasured; (b) vLLM's NVFP4 dense
  `nvfp4_scaled_mm_sm120_kernels.cu` (42.5 MiB MLP weights; builds `Gemm::Arguments` itself, so
  `arguments.scheduler.max_swizzle_size`) — relevant only if prod moves off FlashInfer's `mm_fp4`, whose CUTLASS
  kernel collapsed like unswizzled CUTLASS in finding 99 and would need the fix inside FlashInfer. Steps: extend
  the `_C_swz` standalone (scratchpad `swz/`) to both dispatches, sweep swizzle 1/2/4/8 at the 27B shapes and
  M 4k–32k with `tools/l2sweep27.py`'s shape list, bit-identity vs stock; if the per-channel path collapses
  and recovers, add it to PR #55180 as a second commit under the same activation-slab + L2 gate (bound from
  finding 89's estimate: ~13 % of 8k TTFT, ~22 % at 32k on the 27B). No 27B serve needed for the measurement.
- **27B (prod) — PARKED by the user 2026-09-04 ("27b not now").** Its kernels differ (per-channel FP8
  `CutlassFP8ScaledMMLinearKernel` for attention/GDN projections at 60–120 MiB, FlashInfer CUTLASS
  `mm_fp4` for the NVFP4 MLP at 42.5 MiB — all 2.5–7× the L2) and the bound is larger (~35 % of 8k TTFT,
  ~50 % at 32k/batch 16384, half of that realistic). `tools/l2sweep27.py` + runner `l2sweep27.sh`
  (prod venv, refuses to start if `vllm-qwen38` is up) are ready; the unit was dequeued. Start on request.
- Not queued, by design: full attention (TTFT 30k/7.5k = 3.6× for 3.9× tokens — K/V still fits L2 at
  30k, revisit past 40k), lm_head (last token only at prefill), GDN chunk state (64 KB/head).

### TODO later — downloads (user 2026-09-04: "put downloads in todo for later"; each needs an explicit go)

- primitive-ai `Qwen3.8-Flash-Next-PLE-quant` INT4 g16 sidecars (~32 GB) + the two overlay files — replaces our
  BF16 offload tables + swap dependency; quality check vs the BF16 divergence reference.
- myllmbox `Qwen3.8-Flash-Next-hibrid46` (91 GiB) + bilikaz kit image — A/B vs our FP8-mix + MTP for single-stream
  speed (their 44 tok/s claim) and quality (GDN NVFP4, int3 resident PLE).

### 2026-09-04 — the warm-turn intercept is the 1,600-token align block; upstream state

`hitprobe` (main build, MTP n=3, flag on): hit TTFT = 0.59 s + 0.37 ms × new tokens. The 0.59 s is mostly
the re-prefill of the un-hit tail (prompt mod 1,600 ≈ 1,240 tokens ≈ 0.46 s): vLLM sets the attention block
to 1,600 tokens so one attention page ≥ one GDN state page (`interface.py:918`); `--mamba-block-size` is
overridden in align mode, `MambaDType` has no fp8, so no supported knob changes it. Upstream, checked 09-04:

- **#45702 RFC "Partial Cache Hits for Hybrid Models"** (ZJY0516, 06-15, 23 comments): `hash_block_size` <
  block size + copy-on-write for the partial tail block — exactly this intercept. Scope narrowed (06-23) to
  "re-use the full prompt when its length is not block-aligned" (no SSM checkpoint per fine boundary, so no
  memory blow-up). PoC by Xuan-yi-yan (07-03, DAG-based CoW of SSM state); no PR yet. This is the fix.
- **#52959 RFC "Internal State Checkpoints for Mamba Align Mode"** (ZJY0516, 08-19): one forward instead of
  the split at the last boundary — saves the second full-model pass, not the tail recompute. KDA has it
  (#52789 merged); GDN does not (PR #53614 is Kimi-K3 only).
- **#54458** (net-snix, 08-30, giant pages on GLM-5.3-Flash — we commented 08-30 with the 1,600 number),
  **#45238**, **#40696**, **#53749**, **#50235**: the same mechanism reported five ways (0 % hits below one
  block, misses at exact boundaries, per-request page footprint). Nobody has the warm-turn cost number.
- **#50172** (anuragdutt, GDN `mamba_cache_mode="all"` + MTP on V1, needs rebase): "all" checkpoints every
  block — granularity unchanged (`attn_block_size` still ≥ page ratio), so not a fix for this.

Our contribution: the measured per-turn cost of the granularity on a real agent loop (0.46 s of 1.52 s) and,
if `hitprobe3` confirms it, the boundary-padding workaround for static system prompts — evidence for #45702.

### 2026-09-04 late — union kernel: RFC #55394 open

Feedback until 2026-09-11. Before a PR, in this order: (3) per-request tile boundaries (`query_start_loc` padding), (2) indexer hands over the compact selection via metadata instead of the attribute stash, (1) separate path vs stock-kernel tiling per the maintainers' answer, then a real diff against main. Independent of the answer: dispatch table below 1,024 rows, fuse the build's torch launches.

Implementation plan for the PR (steps 1–8, effort, scope): `notes/upstream/pr-qsa-union-plan.md`.

## Follow-up queued 2026-09-05: block-native R=1 split-K kernel (between #54873 and tile-union)

Review input (2026-09-05, after gau-nernst's RFC reply; the referenced "#54863" is the Intel CI PR — meant is #54873).
#54873 optimises the existing representation (expanded token ids, `valid_count` trims the padded suffix, split-K per
row) and is near its natural limit once every row carries the full 2,048-token selection (its gain falls to ~1.1× at
long prefill). What it does not remove, and what tile-union does: the same K/V block gathered once per neighbouring
row, the M = 16 dot, a page lookup per token instead of per block, ~2,051 int32 index reads per row.

**The intermediate to build and measure — a block-native R=1 kernel:** consume the indexer's compact `[rows, 512]`
block ids (plus the causal tail from the positions), per block resolve the page once (`PAGE % CR == 0`: a compressed
block never straddles a page) and load its CR = 4 consecutive K/V tokens; no sort, no union, no membership matrix, no
tile layout. Expected wins even at R = 1: ~4× smaller selection-buffer traffic, the expansion kernel skipped on
non-MTP-reused layers, one block-table lookup per four tokens, less address arithmetic (the bisect of finding 111 put
pre-resolved addressing at +5–8 % on its own).

| kernel | removes padding | block-native | shares K/V across rows |
| --- | --- | --- | --- |
| old split-K | no | no | no |
| #54873 | yes | no | no |
| **R=1 block-native (to build)** | yes | yes | no |
| tile-union (#55430) | yes | yes | yes |

**Experiment:** the same three captured chunks plus the boundary shapes, against #54873's kernel and the tile-union,
in the `qsa_three_way.py` harness (add the R=1 arm). It attributes the union's 1.45–1.5× between representation /
addressing and cross-row sharing; expectation: a meaningful intermediate, not the union's number at mature contexts
(the ~90 % inter-row overlap stays unused). If the maintainers judge tile-union's ~900 lines too much for ~3 %, the
R=1 kernel is the smaller PR; it also reuses most of tile-union's pack/build machinery minus the sort and membership.
Third item of the same review, already done: the GB10 split-K sweep (finding 120; e2e in `tuchoice`).
Order after the current queue (choice run, boundary run): (1) R=1 kernel in the three-way harness, (2) if it holds,
a server A/B, (3) decide which of the two goes forward as the PR.

