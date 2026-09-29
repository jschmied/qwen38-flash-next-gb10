# Open work, ranked

Cleaned 2026-09-24. Everything closed, superseded or historical moved verbatim to
[`TODO-archive-2026-09-24.md`](TODO-archive-2026-09-24.md). Findings live in `prefill-investigation.md` /
`determinism-investigation.md`; this file lists only what is still open.

## Current (2026-09-28) — read this first; the sections below are older

**Prod config** (service stopped until the user says "up"): vLLM main `1ea7c63f4` + overlay
(`tools/main/main1ea7-prod-overlay.diff`, regenerated 09-27) incl. RecoverSSM as vllm#58863 with the align boundary fix,
MTP K=5 + probabilistic drafting over the 32k NVFP4 draft slice, `--block-size 1728`, `--prefix-match-unit 64`
(§5t), blazux's tool-call parser guards (`tools/toolguard/`, installed 09-28). Drop-ins in `tools/main/dropins/`.

**HC fusion (`FN_HCFUSE=1`) — DONE 2026-09-28, §5aa:** TTFT −5.5…−7.9 % (8k) / −6.2…−7.8 % (30k), decode null; warm agent
turns only ~−1 % (B null, A −2…−5 %); deterministic 1-ulp drift (replay output differs, reproducible). Needs a quality
check before prod. **IN PROD 2026-09-28** (user: "yes, both goes to prod"; drop-in 55 with F4; prodval3 queued). Upstream PR: not asked. Next byte items: MoE GEMM1 epilogue (~4 %),
deterministic fused finalize (3–5 %).

**F4 (full CUDA graphs, RecoverSSM verify) — DONE 2026-09-28, §5z:** short-context c=1 −1.1…−2.1 %, replay/TTFT −1…−2.5 %,
rest null, outputs identical. The first "8k regression" was nvprobe's random replay nonce (withdrawn). **IN PROD 2026-09-28** (drop-in 55). Un-parks the GPU-side early exit (item 2).

**Open, ranked:**
0. **Weight loading — top priority (user 2026-09-28: "move loading speed on top since it speeds up everything").** **#58868 measured (§5w): main load 505–534 s → 65–66 s, model loading ~10 → ~2 min, output identical.** **Fastload set (#58868 + blazux 16 + 18) IN PROD VENV 2026-09-28** (user: "yes, promote and post"; 2 starts: model loading 605–651 → 56–57 s, same hash (§5w addendum); prodval2 checks the prefilter line); GB10 numbers posted on #58868. Remaining: hand-port blazux's loading patches 15–18 (pread for
   small tensors, expert name index, chunked embedding copy, MTP prefilter) onto 1ea7 — their full set took the main
   load 450–541 s → 35 s. Speeds every A/B start and prod restart.
1. ~~**Decision (user): GDN precision cut.**~~ **DONE 2026-09-28: bf16 SSM state in prod (drop-in 50); validation queued (prodval2).** SWE-bench shows no loss from either (§5u). NVFP4 GDN: −7 % decode, +6 % TTFT;
   bf16 SSM state: no TTFT cost, half the Mamba state. Recommendation: bf16 SSM state for agent work.
2. **GPU-side early exit for the MTP draft loop (confidence stop). — PRICED 2026-09-28 (§5v addendum):** IF nodes alone
   +0.25 / +0.59 % (worthless); with one sync or padded verify rows +1.5…+2.7 % code / +3.6…+5.9 % prose decode, 150–250 LOC,
   medium-high risk. Prerequisite (FULL draft graphs) met since F4. **Sync cost measured: ≤ ~0.8 ms/cycle (§5v add. 2) → one-sync design worth up to +2.7 % code / +5.9 % prose.** Build decision pending (the user's).
   Earlier text: Offline replay (§5v): stop drafting before the
   first draft with p₁ < **0.70** → **+4.1 % code, +7.5 % prose** predicted (flat optimum 0.60–0.75, robust to the cost
   model). Only pays if the draft steps are really skipped: cutting verify rows alone gives +1.8 / +4 %, and a host sync
   per draft step lost 5–6 % (§5b). Design direction: a CUDA-graph conditional (while/if) node around the draft steps, or
   device-side predication, so no host sync; verify batch then varies 2..6 rows at c=1. Overlaps with item 4 (graph
   structure). Validate with a real A/B at τ 0.7 against fixed K=5, code + prose, 2 starts; greedy hashes must match. **Update 09-28:** the sync-free lagged variant (K from the previous round) fails the replay gate (code −0.8…−8 %, prose ≤ +1.9 %, §5v); only an in-round stop pays → CUDA-graph IF nodes over FULL-captured draft steps (high risk, no engine does it) or an nsys check first. Parked behind F4, which gives the FULL draft capture it needs.
4. **F4: full CUDA graphs for the RecoverSSM verify path** (analysis in the lightspeed agenda; est. −1…−2 ms/step;
   also a follow-up PR to #58863).
5. **Nightly forward:** bisect the c=4 −4.5 % / non-reproducible regression on `a9eafde59` (§5s; candidates #58434,
   #58275, #49371) before moving prod to a newer main.
6. **Contamination check v2** (`tools/decblk/decblk2.py`, queued): decode-written cache blocks under MTP + nodrop.
7. F3 hyper-connection tail fusion (~0.5 ms/step); content-built draft vocabulary (TF2, low).

**Taken from TensorFold** (ashhart/TensorFold, reviewed 09-26…09-28; their data is one stream, their own engine):
- a. *Confidence-stopped draft chain* → item 2 (our replay: optimum τ 0.70, not their 0.3).
- b. *Draft vocabulary built from content* (their 79.6k list; a 98k prefix list was 7.7–8.6 % slower for them) → item 7;
  our 32k slice covers 0.1–0.6 pp less text than theirs (§5k), so the A/B is a frequency-built 32k list vs the slice.
- c. *Hyper-connection read-out fused into the MoE GEMMs* (norm into the down projection, mix into the up projection;
  their HC matrices run at 104–148 GB/s vs 200+ for big GEMMs) → F3 (item 7).
- d. **Router logits in fp32** (new): they found bf16 router logits tie at the top-10 cut in ~1/3 of layers, and the
  tie decides the expert. Our Qwen4Exp gate is a plain bf16 `ReplicatedLinear` (`qwen3_next.py:183`); vLLM's
  `GateLinear` can output fp32. Cheap: measure the tie rate on real traffic, then an fp32-gate A/B (quality/determinism).
- e. **Copy windows combined with MTP** (new): when the context holds the text being written (file edits), verify up
  to 7 copied tokens instead of MTP drafts; entry needs 8 matching tokens (shorter matches cost them 5 % on fresh code).
  Our memory `labd-lookup-drafting-lever` lists lookup drafting; theirs is the MTP-combined version. Offline first:
  replay agent trajectories for the share of tokens a copy window would cover.
- f. **Shared expert as the 11th slot of the grouped expert table** (new, low): one MoE launch instead of routed +
  shared; no isolated number from them. Kernel work in the fused MoE path.
- g. *Faster weight loading* (sequential reads instead of mmap faults) → item 3 (#58868 + blazux 16/18).
- h. *Load-time exactness self-check* (2/3/4-row windows must equal 1-row steps, else drafting off) (new, low): a
  guard for our determinism overlays; only if we promise bit-exact drafted output.

**Rejected this round (with data):** adaptive verification (§5p), two-candidate branching incl. the
similar-probability variant (§5v), `--long-prefill-token-threshold` (a stalled client either way).

## Prod state (2026-09-24)

- Unit `vllm-flashnext.service` → `/opt/llm/serve-flashnext.sh`. **Not enabled at boot.**
- Venv `vllm-venv-main1ea7`: the main nightly `1ea7c63f4` plus PR #58439's 4 files plus our prod patches
  (QSADET, DETFIN, LMHEADQ/SCALEINV, LMHEADSCALE, 32k draft-vocab slice). The patches are ported on local branch
  `local/prod-on-main`.
- Checkpoint `qwen38-flash-next-mtpfp4`, running MTP n=3 with `disable_eagle_block_drop` and the 32k draft-vocab
  slice.
- The PLE table is **checkpoint-mapped** (`--engram-config {"checkpoint_mapped":true}`, `FN_PLE_OFFLOAD=0`), so no
  offload worker runs. Swap use fell from ~50 GiB to 5–6 GiB (findings 225–232).
- Drop-ins `20-mtp-promote.conf` and `25-main-mapped.conf`. Revert: `rm 25-main-mapped.conf`, then `daemon-reload`
  and a restart. That returns prod to dev524 with PLE offload.
- **Since 2026-09-24 17:23: NVFP4 draft-head slice** (finding 234). This is drop-in `30-nvfp4-draft-head.conf`
  (`FN_DRAFT_HEAD_NVFP4=1`, `FN_NVFP4_CFG=64,4`) plus the FNNVFP4 patch in the venv (`tools/nvfp4head/`).
  - Verified live: the log line reads `606 -> 45 MiB (NVFP4 slice, FNNVFP4)`; c=1 23.26 ms/tok; output identical.
  - Revert: `rm 30-nvfp4-draft-head.conf`, then `daemon-reload` and a restart. The patch is inert without the env
    lines.
- The unit still grants `CAP_SYS_PTRACE`, which only the offload worker needed. The mapped path does not use it.

## Open — speed levers, ranked by value / cost

1. ~~**Slice the draft head in FP8 or NVFP4, not BF16.**~~ **DONE 2026-09-24 (finding 234): NVFP4 slice −3.4 % c=1, +2.8 % c=4, acceptance unchanged; prod promotion awaits go.** The 32k slice is an exact BF16 dequant (the log says "draft
   head 606 -> 160 MiB per draft step"). FP8 would halve that and NVFP4 would quarter it. Finding 198 measured +7.8%
   from cutting the row count, and this attacks the same projection on bytes. `get_top_tokens` needs
   `torch._scaled_mm` instead of `F.linear`. **Cheapest large win left.**
2. **fp8 KV. UNBLOCKED:** #55557 is an ancestor of `1ea7c63f4`, so prod has it (verified 2026-09-24).
   - Measured ×1.72 KV pool on this box (`fp8-kv.md`).
   - It also doubles the attention block, so check the prefix-cache hit rate on agent turns as well as capacity
     (memory `warm-turn-block-granularity`).
   - Quality gate: logprob divergence, not NIAH.
3. **Capture widths at concurrency.** Try `[4,8,12,16,20,24]` at `SEQS=6` through `FN_CG_SIZES`. Above width 8
   nothing is captured.
   - Owed publicly to MiaAI #19: on 09-07 we said our c=16 number was provisional until this ran.
   - The c=16 bandwidth-ceiling claim is already withdrawn on the Quant Map.
4. **INT8 / MXFP8 on the BF16 leftovers.** These are the shared expert, hc low-rank, router and MTP dense layers:
   16.5 % of kernel time (det-136).
   - The shapes have K=320, which is not 128-divisible, so blockwise FP8 cannot express them. Use MXFP8 (group 32)
     or INT8 per-channel.
   - The four hardcoded `quant_config=None` opt-outs also have to go.
   - The EXL3 field read suggests 7–13 %.
   - Run the shapebench (below) first.
5. ~~**Dynamic stopping / adaptive draft length.**~~ **MEASURED 2026-09-24 (finding 235): n=3 thr 0.3 −1.7 % c=1 / +2.5 % c=4, n=4 loses; not in prod; stacking with the NVFP4 head needs a small patch + A/B.**
   - Depth alone loses: finding 155 measured k=4 at −3.4 %.
   - The depth band 5..8 needs the QSA-ring widening from our PR #54912. It is patched locally already (det-204).
   - The plumbing is issue #57608 (a host hook between proposal and verify).
5b. **DONE 2026-09-28 (§5aa): prefill intermediates ~15–20 % of TTFT; #1 HC fusion ≈ −7 % TTFT (up-GEMM gate-mix epilogue + rrms-only combine_norm), #2 MoE GEMM1 epilogue ≈ 4 %, #3 fused finalize 3–5 % (determinism trade). Decode < 1 %.** Original item: Intermediate-byte ledger (user 2026-09-28: "do we have fusable kernels where a fusion would lower bytes
   read/written? I think we checked only for launch overhead").** §4m/§5f priced fusions by launches and gaps, and §5f by
   avoidable work; nobody has listed producer→consumer tensors that round-trip DRAM. GB10's ncu has no DRAM byte
   counters (finding 144), so compute bytes from shapes per kernel, and decide L2 vs DRAM by size against the 24 MiB L2.
   Two regimes: **decode** (M = 6 rows: activations are KBs, weights dominate; only state-sized intermediates count:
   RecoverSSM commit → next verify re-reads the checkpoint, ~54 MiB/step at bf16 ≈ 0.25 ms; QSA split-K fp32 partials,
   6 MiB/layer, probably L2-resident) and **prefill** (M = thousands: SwiGLU + fp4 quant into MoE GEMM1's epilogue
   ≈ 1.1 of 13.5 ms/layer at 7.5k, item 6; FP8 act-quant ×96 re-reading bf16 activations; HC stream reads/writes).
   Prefill first: agent speed is TTFT-bound.
6. **MoE epilogue fusion** (findings 144/145). **DONE 2026-09-29 (§5ag): Triton prefill MoE, bit-identical to FlashInfer, TTFT −3.8…−4.3 % at 30k; proposed, not in prod.** History below. **Scoped 2026-09-28 (plan, agenda item 5):** the prize is ~4 % of TTFT
   (§5aa: GEMM1 output + bf16 activation round trips, ~264 MB per layer-chunk). Where a fused GLU exists today:
   - FlashInfer 0.6.18's SM100 "mega" CuTe-DSL MoE (`moe_nvfp4_swapab`, `runner_fc12`): fused fc1/GLU/fc2, **SM100 only**.
   - **b12x's gated-optimized kernel** (`fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`): runs on sm_12x but
     is gated off for us by `_GATED_OPTIMIZED_RETAINED_SLICES = 4` (× 128 = max intermediate 512; ours is 640 = 5
     slices). Every other bound fits (hidden 2,560 ≤ 16,384, top-10 ≤ 16, tile 128×128, sf_vec 16). Finding 192 saw b12x
     run its *generic* kernel for exactly this reason.
   - CUTLASS SM120 grouped GEMM (today's `device_kernel` + separate `doActivation`): an EVT epilogue for SwiGLU + fp4
     quant is the general fix and the most work (12-min FlashInfer rebuilds, `tools/moe_swz.py` harness).
   **Step 1 CLOSED 2026-09-28 (§5aa):** the b12x gated kernel needs 128×128 MMA tiles, which b12x never picks for 512
   experts at ≤ 4,096-token chunks (it picks 32×128 / 64×128); b12x is also run-to-run nondeterministic (0.65 % rel L2).
   Only the CUTLASS EVT route remains. **Scoped 2026-09-28:** FlashInfer 0.6.18's TMA warp-specialized MoE declares
   `EpilogueFusion::GATED_ACTIVATION` but throws "Unimplemented fusion" for it (moe_gemm_template_dispatch.h:882); the
   only fused gated path is Ampere bf16/fp16 (`supportsFusedGatedActivation`), not FP4. No upstream SM120 implementation
   (TRT-LLM's FC12 / FlashInfer's mega fused kernels are SM100/Rubin CuTe-DSL). Building it = a new SM120 EVT epilogue:
   SwiGLU over interleaved gate/up N-tiles (reorder w13 rows offline) + NVFP4 output with per-16 block scales, JIT
   generation in `jit/gemm/cutlass/generate_kernels.py`, 12-min rebuilds. Multi-day; prize ~4 % TTFT. Needs the user's
   go before starting.
   **Design template found (2026-09-28):** FlashInfer's `csrc/cute_sm120_mxfp8_groupwise/sm120_fused_moe/` (FP8/MXFP8, SM120)
   computes each output tile's gate and up halves in the same CTA (`mB_gate`/`mA_gate` at offset N, two accumulators) and
   applies the activation in-kernel: the dual-tile answer to the N→N/2 problem an EVT epilogue cannot solve. Porting it to
   our NVFP4 experts needs the SM120 4-bit block-scaled MMA (16-element blocks, e4m3 scales; CUTLASS example 79a/79b
   territory) and an FP4 output with block scales in GEMM2's swizzled SF layout. MXFP8 experts are not an option (twice
   the expert bytes at decode). Multi-day kernel port; stopped per the agenda's blocker rule, awaiting the user's go.
   **Order (original):** (1) clone venv: RETAINED_SLICES 4 → 5, compile check, then a one-layer b12x-gated vs cutlass standalone
   (correctness vs cutlass output, time at M = 3,456 × top-10) — cheapest, may simply not fit in shared memory;
   (2) only if (1) fails, the CUTLASS EVT route. Determinism: b12x's finalize path must be checked against our
   bit-stable-finalize requirement before any server arm. This is the 36.4 % bucket at the DRAM floor: the biggest kernel prize
   and the biggest effort.
7. **65k vs 32k draft vocab.** One arm; cheap disconfirmation, and 32k is expected to hold.
8. **Finish the MTP re-measurement** (`mtp-remeasure-plan.md`). The Quant Map page flags its depth curve as under
   re-measurement.
10. **Levers from the myllmbox v4 recipe (the-field, 2026-09-29), one A/B each on the prod config, after the §5ah
    quality screen:** (a) `rejection_sample_method: block` with our probabilistic drafts (acceptance); (b)
    `vm.compaction_proactiveness=0` vs our 20 (host sysctl, reversible; they report a 4–5 s stall every ~37 s from
    migrating GPU-mapped pages); (c) Marlin MoE at decode vs FlashInfer CUTLASS (our Triton prefill MoE reads
    FlashInfer's layout, so this needs a split); (d) NVFP4 W4A16 target lm_head (0.33 vs ~0.64 GB per cycle);
    (e) vllm#58449 fused draft metadata vs the confidence stop's per-step overhead.
9. **Cherry-pick two merged upstream PRs that run in our decode path, one A/B each** (added 2026-09-29, user: "record
   as todo"; not a merge of main, per the cherry-pick rule):
   - **#58957** (NVIDIA HC down projection + SiLU, CuTe DSL, decode M ≤ 48). First: its tests and a compile on sm_121
     (only tested on GB300). Then a width dispatch against our HC fusion in the same file (`_down_and_inject`: ≤ 48
     theirs, ≥ 128 ours, 49–127 unfused), then a decode A/B on the clone venv, 2 starts per arm. Expected on GB10:
     +0.5…1.5 % decode (same weight bytes; saves a launch and the SiLU pass), vs their −2…−4 % TPOT on GB300 TP4.
   - **#58114** ("Reduce PLE metadata construction overhead"): same procedure, per-step CPU/launch overhead on our
     path.
   - Not taken: #58706 (ROCm only, `amd/` files) and #53909 (standalone kernels, not wired into the model).
   - Moving prod's whole base to current main would bring all of these, but it means porting every overlay (RecoverSSM,
     fastload, HC fusion, F4, the §5ag candidates); separate job, not mixed with these A/Bs. See the-field "Upstream
     HC fusion work vs ours".

## ~~FULL_DECODE_ONLY decode graphs~~ — MEASURED 2026-09-24 (finding 237): decode null, agent turns 3–4 % slower; keep PIECEWISE. `be7a84fe4` pushed to #58439.

## Needs prod DOWN

- **Shapebench of the hyper-connection shapes** (`tools/shapebench.py`). Question: does FP8 `_scaled_mm` beat cuBLAS
  BF16 at (10240, 320) and (336, 10240) for M=1..8?
  - It OOM'd beside a live prod on 09-22: `MemAvailable` does not show GPU-allocatable memory next to a util=0.90
    server.
  - **Do not shrink the ~300 MB rotation.** That rotation is what defeats L2.
  - It gates lever 4.
- **smgates rescan on main.** Scanner v2 is `/opt/llm/runners/smgates.py`. The last scan was on dev401; rerun it
  after every venv bump. This one needs no GPU and can run anytime.

## PLE checkpoint-mapped PR (#58439) — follow-ups

- **Load-time gap.** Expert files load about 15 % slower with offload off; the mechanism is unknown. Next step: a
  py-spy profile of the loader, one start per arm.
- **Not coverable here:**
  - TP>1 (one GB10);
  - the real `reload_weights` lifecycle, which the unit tests cover but no integration test does.
  - Say so in the PR if a reviewer asks.

- **2026-09-25: cold-start decode wait (FN_PLE_SYNCTOUCH + C populate helper), speed-of-light 4f–4h.** Cold −2.2 ms/step
  (−3.7 %, outputs identical), but an unconditional wait costs +2.4 ms/step warm → auto mode (wait only while ≥12
  faults/step) is built, not yet validated. Plan: validate (2–3 starts, cold gain + warm neutral) → follow-up PR on top
  of #58439 (bounded decode wait + GIL-free page fill as a `csrc/` CPU op), NOT into #58439.
- **LATER (user 2026-09-25): head-to-head vs #54129** (CPU gather + H2D) on GB10 before claiming "best PLE loader":
  #54129 overlay vs ours stock vs ours+auto, cold + warm pass, 2 starts/arm, KV 4 GiB (~3 h, prod down).
- **With a go:** one line in #58439 (open question 2): the prefetch rescues cold prefill (88 s → 1.6 s) but can
  never win in decode (ids known only when the previous step ends) → a fresh server pays ~4 ms/step of serial GPU
  faults (~11 %) until warm.

## Upstream — watch, and never reply without a go

| # | ours? | state (2026-09-24) |
|---|---|---|
| #58439 PLE checkpoint-mapped backend | PR | open, bot comments only |
| #58441 PinnedHost side-stream race | issue | fixed by #58489 (Juntian777, open) |
| #54076 prefix-cache arm | owed → **delivered** 09-23 (comment v2) | watch for wickist / MaCoredroid |
| #55122 det top-k | PR | open. **Perf measured on the head, det-235:** the kernel is 4–26 % faster than stock on GPU time and indistinguishable end to end; it is 3–4 % faster than exact `torch.topk` in 30k TTFT. Posted 09-24 (issuecomment-5809971218); watch for replies. |
| #54912 QSA ring widening | PR | open, no human review since 09-02 |
| #38315 FLA fused kkt+solve | not ours | open; our `pr-fla-fused-kkt-solve.md` stays unopened as a duplicate |

**Sweep 2026-09-24:**
- **#58489** was approved by ZJY0516, and CI is running.
- **#58439:** hclsys independently confirmed the gate attributes on a second Spark; the PR is still waiting for
  maintainer review.
- **#56964** now carries our inverted-check design, so it would retire the prod genfix overlay once merged.
- **#58157:** hclsys ran our test on GB10 (2 passed). They also flag an interaction with their #57512 (on SM12x
  every fp32-scale checkpoint would warn) and suggest stating it in the description, which needs a go.
- **New PRs to watch:**
  - #58449, fused QSA draft-metadata updates: **measured 2026-09-24 (finding 233): correct on GB10, speed null,
    outputs bit-identical**; not carried in prod;
  - #58040, a QSA metadata clamp for graph-padded offsets;
  - #58114, PLE metadata overhead;
  - #58300, peakcrosser7's Qwen4Exp cleanup, which touches `ngram_embedding.py` and so will conflict with #58439;
  - #58207, the hybrid KV group-size heuristic;
  - #58068, fixed-width indexer logits;
  - #58310, the Engram host-memory check.
- **New issues:**
  - #58303, Mamba+EAGLE dense retention going back to 0 % prefix reuse under interleaved load, which is our
    MTP+prefix-cache setup;
  - #58422, a TP1 engine wedge on a Qwen3.8 GDN hybrid with MTP4 on SM120;
  - #58080, the MTP draft not inheriting `--hf-overrides`.
- **Main `b44895cf9`** (109 commits past prod):
  - FlashInfer **0.7.0** (#58069): the GDN call now passes `backend="flashinfer"`, and jit-cache is `+cu134`. Audit
    the wheel for sm_121 before any bump.
  - #57176, per-token NVFP4 MoE backend selection;
  - #49845, the KV block size chosen to suit every attention backend;
  - #58459, adaptive `--long-prefill-token-threshold`.

**vllm#53912 / #57128** (prefix cache + spec decode + `disable_eagle_block_drop` poisoning cached Mamba state): our
exact config family. **Tested 2026-09-24 (det-236): not reproduced on prod.** 15 low-acceptance cache reads into
decode-written blocks, 0 divergent. **CLOSED 2026-09-28 (§5x): at prod's default retention interval 0 a decode-written state is never served
(decblk5, 16/16 hit 0); with interval 3456 it is served and did not contaminate (decblk4: 4 vs 2 of 16, late drift).** Keep watching; a comment with the MTP counter-datapoint could help, but needs a
go.

**Stale drafts** from 09-08/09, measured on dev401/fnmain2: `comment-54521-zc502-isolation.md`,
`comment-54521-tcorrupt.md`, `comment-miaai-19-cudagraph-widths.md`. Re-check them against the current stack before
any post, or drop them.

## Parked — each needs a user decision

- **Enable the prod unit at boot.**
- **Job 90** (`ishare2`), in `~/qwen-night/jobs/parked/`.
- **Devanagari U+093E → U+094B.** After det-212..219 every cheap hypothesis is dead (GDN kernel, MTP, FlashInfer,
  tile-union).
  - On the 12-prompt probe fnmain3 is 8/12 vs fnmain2 10/12 (per prompt Fisher p = 0.64; the old 0.318 counted
    byte-identical repeats). det-221 ties hi-01 to #55272 (compile removal); hi-05 is unexplained. Checkpoint
    attribution was never tested (no BF16 reference).
  - Remaining bisect rungs are hand-ports, hours each (det-216).
  - The stock RadixArk checkpoint is no longer on disk.
- **`--mamba-ssm-cache-dtype bfloat16`** (finding 153). Total agent-turn TTFT improves −9.6 %, but 127/2,504 modal
  top-1 predictions change. Ship only after a task-level eval.
- **SWE-bench Multilingual-28 (Go/Java).** The x86 box 10.0.0.8 answers ping again (2026-09-24), so the task is
  unblocked. Recipe: memory `swebench-x86-eval-recipe`, tunnel `-R 8080:127.0.0.1:8092`.
- **NVFP4 PLE table** (#56273, or the provsalt/starkweather checkpoints). With the mapped table this no longer buys
  residency, only less page cache and fewer SSD reads. Low value; it needs a download go.
- **27B FlashInfer large KV pages** (TEST B in the archive). This is a real A/B on the 27B, which is not this repo's
  model.
- **#55122 follow-ups:** the H100 filtered-path steps 2/3, and `perf/gemm-launch-hwinfo` (hygiene only).
- **A second QSA decode backend.** Only this would make #48162 do anything.
- **Batch-shape non-invariance**: 416 flips at 1,999 tokens. Matters only if batch-invariant evals do.
- **Disk:** 260 GB free. `qwen38-flash-next-mtpfp4d` was rejected in finding 216 and is a deletion candidate. Ask
  first.

## Watch — field

- **PixelML DFlash drafter** for `nvidia/…-NVFP4`. Weights have been up since 2026-09-10 (848 downloads).
  - Their README calls it "a code wash" and measured it in eager mode.
  - Before any download, check whether it transfers to RadixArk-based checkpoints, and measure on agent traffic
    with cudagraphs on.
- **SGLang on SM121.** #36497 and #36556 are unmerged. Watch; do not attempt.

## Settled — do not re-open without new evidence

- **Hyper-connections**: three interventions, all null; latency-bound at ~78 % of roofline.
- **NVFP4 KV**: three GB10 measurements, an MTP-acceptance penalty and silent failure. (FP8 KV is lever 2.)
- **MTP k=1** is strictly dominated. **MTP k=4** measured −3.4 % (finding 155), and PixelML agrees.
- **Quantizing the MTP module body** (finding 216): not faster, no attributable KV gain.
- **`--max-num-batched-tokens` 8192**: reversed at n=3; stays at 4096.
- **`--max-num-seqs` 16 → 64**: null. The "bandwidth ceiling" explanation for c=16 is **withdrawn**; see lever 3.
- **MoE backend**: `auto` picks `FLASHINFER_CUTLASS`. b12x works with #57946 plus the generalized #56964 (memory
  `b12x-50189-root-cause`).
- **FlashInfer AOT prebake**: the wheel already ships the `.so` files. The old "do not bump past 0.6.17" warning was
  refuted in det-208.
- **Lowering `gpu-memory-utilization`**: refuted.
- **Index sharing** (finding 159), **CPU idle states** (156), **FLA fused kkt+solve on agent turns** (154): all
  null.
- **Async scheduling + MTP**: bit-identical output (158).
- **#55390**: merged and in `1ea7c63f4`. It suppresses a false warning; reuse was already 71.3 % without it.

## Standing rules earned the hard way

- **Noise floor**: 6.9 % for decode, **±20 % for prefill** (n ≥ 3).
- **Verify the lever at the shape level before building** (`tools/shapebench.py`).
- **A serving config has capabilities, not just speed.** Probe tool calls and a long generation before benchmarking.
- **Prove a kernel ran** by its effect. For example, `grep -c <lib>.so /proc/<worker>/maps`; not a log print and not
  `/proc/<pid>/environ` (memory `proc-environ-invalid-for-vllm`).
- **Clear `VLLM_CACHE_ROOT` and `TORCHINDUCTOR_CACHE_DIR`** for source-level patches.
- **A gate must ask the consumer's question**, and it must test a captured string, not a pipeline's exit status.
- **`max_tokens` is a ceiling.** Use `ignore_eos` to force a length, and detect empty output by
  `finish_reason: "length"`.
- **An accept-length pinned at its maximum is a corruption signature.**
- **Acceptance is a speed statistic, not a quality signal.** A 1-ulp kernel change moves it ±10 pp.
- **Rank warm-turn levers on the paired per-turn total**, never the median.
- **Treat a field number as a hypothesis about a mechanism**, not as an expected effect size.
- **Secret scan including Modal tokens:**
  `git grep -nI -E "sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|\b(ak|as)-[A-Za-z0-9]{20,}|develop8\.|BEGIN [A-Z ]*PRIVATE KEY" -- .`
- **Anything a server must read goes in `/opt/llm/runners`**, world-readable. `/tmp/claude-1000` is private to
  jschmied.
