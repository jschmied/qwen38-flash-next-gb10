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

## Queued experiment: quantize the PLE table (2026-08-31)

**Why it outranks the speed levers.** The PLE n-gram table is **51.2 GB FP8 of a ~135 GB
checkpoint** on a 128 GB box. At NVFP4 it would be ~25.6 GB — which does not make the CPU-offload
path *faster*, it potentially makes the whole 1,400-line offload subsystem **unnecessary**. That is
a structural simplification, not a tuning win, and it removes our single largest source of local
patches and startup fragility ([[fnext-venv-ple-backport]] exists only because of offload).

**vLLM cannot do this today, verified in our tree.** `modelopt.py` `get_quant_method` dispatches on
`(LinearBase, ParallelLMHead)`, then `RoutedExperts`, then `return None`. `ParallelLMHead` is a
*subclass* of `VocabParallelEmbedding`, so a **plain embedding never matches any branch** — there is
no embedding-quantization path at all. SGLang has `ModelOptNvFp4EmbeddingMethod`.

**Do not build a checkpoint first.** `starkweatherdigital` already published an NVFP4 PLE
(102.4 → 28.80 GB, same E2M1 + per-16 FP8-block-scale layout as the experts) behind a ~130-line
loader patch and `VLLM_PLE_NVFP4=1`. Testing theirs is hours; building ours is days.

**Order of work, cheapest disqualifier first:**
1. **Cosine against BF16 on the PLE table, offline, no server.** Our FP8 PLE matches BF16 at
   0.999635. NVFP4 on a table that feeds *every* layer is a far larger step than FP8 was, and this
   is the measurement that kills the idea cheaply if it is going to die.
2. Only if that holds: load their checkpoint, confirm it serves without the offload worker.
3. Only then: quality gate (NIAH is not sufficient — it passed at 71.85 ms/tok, it cannot see
   degradation), then decode/TTFT/ms-tok against the current config.

⚠️ NIAH cannot gate this. Thirteen consecutive 5/5 passes across every config today, including the
worst-performing one, means it discriminates nothing. A PLE quality regression needs a real
comparison — logprob divergence or a task score — before any number is believed.

Related: [[speculation-costs-kv-pool]] (the other place block geometry eats memory),
[[evidence-standard]].

## Later: is vllm#48162 mergeable onto our branch — and would it do anything?

**The mergeability test is the easy half and the wrong question to start with.**

#48162 ("[Attention] Batch-level prefill/decode attention backend routing", +2037/−78, 38 files,
open since 2026-07-09) adds `--attention-decode-backend` / `--attention-prefill-backend`. Merging it
onto our tree is a mechanical question worth an afternoon.

**But it would be inert for us as things stand.** `nvidia/qsa.py:341` hard-wires
`self.attn_backend = Qwen3_8FlashNextQSAFlashAttentionBackend`, returned verbatim by
`get_attn_backend()`. vLLM's own comment: *"models that hard-wire their backend never consult it"*
(`attn_utils.py:190`). Our 12 QSA layers are the only full-attention layers in the model, so the
flag reaches nothing.

**The real prerequisite: what would we switch decode TO?** We have exactly one QSA implementation.
A per-phase selector with one option per phase does nothing. Candidates, both problematic:

- **FlashInfer block-sparse decode** — `trtllm_batch_decode_with_kv_cache(...,
  enable_block_sparse_attention=True)` is already in our 0.6.17, takes per-KV-head page indices, and
  sgl#36558's reporter verified the kernel numerically correct on SM121 (max abs diff 4.70e-04).
  ⚠️ But `decode.py:3260` selects trtllm-gen only when `capability[0] == 10`, so GB10 (major 12)
  auto-selects `xqa` and block-sparse then raises; and our own [[failure-modes]] records trtllm-gen
  decode kernels **silently emitting garbage on SM121** (`!!!!` forever after a correct first
  token). Two field reports directly contradict each other. Gate on numerics against the current
  kernel, never on "it started".
- **sglang#36845's Triton QSA kernel** — closed separately: 0.5% of GPU kernel time here, and it
  expects flat packed varlen KV where we page.

**Order if this is ever picked up:** (1) establish that a second decode backend exists and is
numerically correct on sm_121 — that is the whole risk; (2) un-hardwire `qsa.py:341`; (3) only then
care whether #48162 merges. Doing it in the reverse order spends the effort before learning there is
nothing to select between.

⚠️ **It also fights tonight's other lever.** `attn_utils.py:174-176` takes the **minimum** cudagraph
support across an attention group, so adding a weaker second backend would silently downgrade our
capture mode — the opposite of what the hyper-connection work is trying to do.

- **Batch invariance is unavailable on this architecture** (`notes/batch-invariance-unavailable.md`):
  no mamba/GDN backend implements `supports_batch_invariance()`, and 36/48 layers are linear
  attention. Worth raising upstream as a gap once the determinism root cause is known.
- **Layer bisection** (`layerhash_patch.py`, queued as unit `bisect`): first differing layer over
  three identical prefills. layers 0-1 same + 2 onward differing implicates the PLE; layer 0
  already differing puts it below the model (embeddings / first GEMM / CUDA).
- **Acceptance correlation** (queued as unit `accepcorr`): 5 starts of MTP n=5 pairing `ms/tok`
  with `mean_accept_len`, to test whether acceptance is the channel turning divergence into a
  1.83x throughput spread. See `prefill-divergence.md`.

- **[queued 2026-09-05] Block-native R=1 split-K QSA kernel** — compact block ids in, page resolved once per block,
  4 tokens loaded per block, no sort/union/membership; benchmark against #54873's kernel and tile-union on the captured
  chunks + boundary shapes to attribute the union's gain (representation vs cross-row sharing). Design and experiment in
  `notes/prefill-plan.md` (follow-up section). Candidate for the smaller upstream PR.

- **[parked 2026-09-05] `nvidia/Qwen3.8-Flash-Next-NVFP4`** — same bits as RadixArk on the experts (NVFP4 g16, MSE scales),
  BF16 dense (no FP8-dense lever), FP8 PLE/MTP byte-identical to Qwen FP8; their FP8-vs-NVFP4 table is within ±1 point on
  nine benchmarks. Decision: ignore for now — no expected intelligence difference; only a long-generation divergence check
  would tell, and that costs 74 GiB + a day. Revisit only on a concrete quality problem with RadixArk's experts.
- **[2026-09-05] MTP multi-prefill output corruption — ROOT-CAUSED AND FIXED** (findings 126–131): strided `state_indices_p[:, 0]`
  view (1+n_spec columns under spec config) read with unit stride by the PLE short-conv kernels → rows ≥1 of a prefill-only
  step write their conv state into request 0's checkpoint blocks. Fix = stride-aware kernels (branch
  `fix/mamba-prefill-state-indices-contiguous`, 20/20 clean). Upstream fix = vllm#55375 (ours #55467 closed as duplicate; evidence + decode-mode test offered there);
  overlay the one-file fix onto the prod venv before re-enabling MTP; re-measure the MTP c≥2 cells (123/126) on the fixed build.
- **[2026-09-05] GB10 split-K table PR — DROPPED** (finding 132): kernel gains real (up to 2.25× on 4×4) but 0.4 % of a step; no server-level effect at three starts. Keep the stacked branch as reference only.

## Actionable after the 2026-09-06 determinism closure (ranked; state of the box: idle, prod venv carries the four fixes)

1. ~~**Our upstream PR vllm#55122 carries the v2.3 guard bug**~~ — **DONE 2026-09-06 20:3x** (upstream log 57). Pushed `afd92810`
   (the 09-03 CodeRabbit round, which had never reached the PR) + `c564e5c1` (guard conditional on the cooperative path,
   `test_persistent_topk_short_rows`, 33 cases). 134 passed against `_C_det` v2.4; 24/33 fail against v2.3, so the test catches it.
   Comment posted (upstream log 58). CI: `pre-run-check` fails on vLLM's label policy (no `ready`/`verified` label, 0 merged PRs
   by this author) — not a code signal, and the workflow says not to ask for the label; nothing else runs until then.
2. **Prod MTP recipe lacks `disable_eagle_block_drop`**: only the `FN_SPEC_METHOD` path honours `FN_SPEC_NODROP`; the `FN_MTP`
   shortcut does not. Measured −26 % per warm turn with MTP + prefix cache (3 starts, 2026-09-04). Extend the `FN_MTP` path,
   default it on, one validation start — prod change, needs the go.
3. **Re-measure everything taken one PLE step behind (2026-09-03 .. 09-06)**: MTP acceptance on agent traffic (41–50 %), the
   draft-vocab +6 % (det-135/137), the warm-turn cost (finding 141), any SWE figures. One dvrate-style cell, 3 starts, fixed stack.
4. **Draft-vocab 32k slice into prod**: decide after 3. **Not publishable until then** (asked 2026-09-07): det-135/137 are
   inside the one-PLE-step-behind window AND det-137 is only two starts per arm against our own three-start rule. Re-measure
   on the fixed stack first, then it is worth a post — +6 % single-stream decode from a 32k draft-vocab slice is a real result.
3b. **QUEUED 2026-09-06 (user: "queue re-measurement and page update for later on the prod stack with all fixes"): the MTP
   cells + the published Quant Map.** Full spec in `notes/mtp-remeasure-plan.md` — 5 cell groups, ~51 starts, an overnight job;
   the prod venv already carries every fix it needs (#55375 stride, PLE semaphore, det overlays, `disable_eagle_block_drop`).
   The page keeps its affected cells until this runs; that is the accepted trade. **Group 0 added 2026-09-06:** every
   tok/s on the page is from the PREVIEW build (`fnext`, 0.1.dev20073) while prod serves the MAIN build (`fnmain2`,
   0.28.1rc1.dev401) — the non-MTP ladder is stale for that reason alone, and no preview-vs-main decode comparison exists.
5. **Upstream watch**: peakcrosser7's response on fork PR #13; when #53899 merges into main, re-port the overlay and open the
   semaphore fix against main; #38315 auto-closes ~2026-09-10 → then open our FLA kkt+solve PR (branch ready); ZC502 on #54521.
6. **Batch-shape non-invariance** (identical prompts in one batch differ: 416 flips at 1,999 tokens): separate lever, only if
   batch-invariant evals matter.
7. **Disk**: /opt 56 GB free; today's per-arm caches < 1 GB, something else is large — check before the next model pull.

## Work queue as of 2026-09-07 (user: "higher prio has work on our own findings and PRs, run when idle")

### HIGH — our own findings and PRs

- **After ANY kernel change, rebuild from `patches/kernel-det/` before trusting it** (2026-09-07,
  upstream log 81). The published launcher had drifted from the tested build and HEAD did not
  compile; blazux/qwen3.8-Flash-DGX fetches those exact files by pinned SHA + sha256. `cmp` each file
  against `/opt/llm/runners/kdet_build/`, then run `build_det.py` over a copy of the directory the way
  a consumer does. Open: blazux invites PRs for their #5 (CI for the PLE test) and a guard-counter
  metric (#8) — both small, ours to pick up if we want them.

- **Watchdog thread list is missing #53670** (found 2026-09-07, upstream log 79). The hourly watchdog
  checks 13 issue numbers; #53670 is not among them although `upstream-post`'s venue table lists it and
  we post there. Two human comments went unseen for eight hours. **Add #53670**, and diff the whole
  watchdog list against the venue table + our posting log for other gaps before the next tick that
  matters. Also watch #54360 (prefix-cache hits to ZERO on hybrid GDN) — named in that thread as a
  separate, larger failure than the trailing block, and we have not looked at it.

- **Filtered-path speed on ≥128 KiB parts (H100/A100) — three steps, in this order.** Context: det-153
  (fix 1 shipped, `3e399815` + `995cd99f`), det-154 (fix 2 built and rejected), and the H100/A100
  measurements in `bench/h100-filtered`. The path is 1.1–2.7× upstream, worst at n=65,536. Bundle and
  Modal harness are built and validated; each measurement round is minutes and cents.

  1. ~~**Routing — is `rows > 32 → FilteredTopK` stale?**~~ **CLOSED 2026-09-07, it is sound** (det-156,
     3 starts each arm; persistent wins 3 of 45 cells, all at 64x65,536, and loses at 128/256 rows even
     there — an occupancy artifact, not a dispatch bug). Upstream comment edited to say so. Skip to 3.
     Original text: **is `rows > 32 → FilteredTopK` stale?** `FilteredTopKUnifiedKernel` launches one
     CTA per row, so 64 rows on a 132-SM H100 idles more than half the machine, while the persistent
     path splits a row across CTAs. Our own scaling is sublinear 64→256 rows on both arms, which fits.
     **Prerequisite, do this before any benchmark:** verify the workspace and `RadixRowState` sizing
     support `num_rows > 32` (`kDetMaxCtasPerGroup` = 64, `num_groups` from occupancy) — that path has
     never been entered with more than 32 rows on a ≥128 KiB part. Then force both paths at rows
     {48, 64, 128, 256} × n {16k, 20k, 40k, 65k} × k {512, 2048} on H100 and find the crossover.
     **Counter-evidence to respect:** at n=16,384 our filtered path already beats our persistent path
     (17.9 µs at 33 rows vs 21.9 at 32), so any fix is shape-dependent on `n`, never on row count
     alone. If persistent wins at the long rows this is a dispatch condition — small, in scope for
     #55122. Question posted to the PR 2026-09-07 (upstream log 78); a maintainer answer on the
     history of the 32 may settle it without measuring.

  2. **Templated survivor compaction — only if 1 does not carry it.** The rejected patch is at
     `notes/data/fix2-survivor-compaction.patch`; it delivers (H100 65k 2.70 → 2.25) but merely
     compiling the runtime-disabled branch costs GB10 decode 9–15 % at identical REG/SHARED. Fix is a
     compile-time specialisation — `det_select_row<bool Compact>`, with the compacting variant
     referenced only by a separate filtered-kernel instantiation so it cannot perturb the persistent
     kernel; the host already knows `cached = fixed + n*4 <= cap` and can pick before launch.
     **Known limit, and the reason this is second not first:** pass-0 compaction is weak on narrow QSA
     score distributions where many keys share the high FP32 byte — the threshold bin can be a large
     fraction of the row, so it would rarely arm on exactly the #51782 inputs that motivated the PR.
     Ceiling is ~2.25×, so this does not solve the path either.

  3. **The real fix: an exact filtered algorithm — separate PR, not #55122.** n=20,000 on H100 is
     *cached* and still 2.2× (20.3 vs 9.2 µs), so the remaining cost is repeated histogram/atomic
     work, not memory traffic — the four-pass generic radix is the floor. Design: 4096-bin coarse
     histogram (12 key bits) → one more full-row scan that emits definitely-selected indices and
     threshold-bin candidates **both in index order via the packed `BlockScan`, never `atomicAdd`
     slots** → exact refinement over the candidates only → merge two ascending runs of at most K.
     Two full-row scans plus O(K), against today's four radix passes plus a full-row emission.
     **The exactness comes from the fallback, not the histogram width:** `threshold_bin_count` is
     known before collection, so `count <= capacity` takes the fast path and anything else falls back
     to `det_select_row`. That is the distinction from #53287, which widens the histogram to make
     overflow rarer rather than harmless — worth stating that way if this is ever written up.
     Plausible ~1.1–1.4×; do not promise it before measuring. Land #55122 on correctness first unless
     a reviewer blocks on cost.


- **PR #55122 (det top-k):** ~~v2.7 not pushed~~ **PUSHED 2026-09-07 as `b8d09ecb`** (upstream log 66). Still owed:
  ~~the cost table~~, ~~the tkprd reply~~, ~~the local-review items~~ — all done and **pushed 2026-09-07**
  (log 72; 13 commits, body carries the risks). **Still open:** prod runs v2.4 while the branch is at v2.8 (`/opt/llm/kernel-det/_C_det.so`);
  no human review yet and `pre-run-check` blocks CI on the label gate; **and two contacts drafted but NOT
  approved** — ~~a post on #51782~~ (POSTED, log 69; the third path is bypassed, verified) and a single ping to
  ywang96, who merged our #55180 — still not approved. **Held, not pushed:** the 2048-bin decode-path removal (275 lines, tested 210/210 + 142) sits on
  `wip/topk-remove-decode-path`; pushed once by mistake and reverted on request, awaiting a go.
  **Also found:** #53287 (LopezCastroRoberto, open since
  08-21) argues the opposite conclusion — no measurable accuracy regression in MAIN — so it is a related open
  PR with a different stance, not a duplicate. And `histogram_2048_topk` is now dead code on our branch, the
  same way `det_sort_row` was; a mechanical follow-up commit, not folded in. `/csrc/libtorch_stable` has no CODEOWNERS entry, which is why this PR has no owner. Whole grid 1.00–2.45× (was 1.25–4.31×),
  14 of 43 cells at or below stock, 210/210 + the PR's own 134 pytest cases, bit-identical throughout.
  Needs: push to the branch, rewrite the cost table a third time, reply to gau-nernst. Prod still runs v2.4.
- **PR #55661 (swizzle gate):** blocked on the `swzab2` server A/B. Decision rule from the user:
  ≥1–2 % TTFT at 2.5–5k with no losses → defend; <1 % → close. **Independently of the A/B, drop the
  small-M island** — it buys 18.7 pp on the tuned set and *exactly zero* out of sample (slab-only and
  island both score 236.3 pp on 88 held-out cells), which is the reviewer's "overfitting" point, proven.
- **`perf/gemm-launch-hwinfo` (local, unpushed):** KernelHardwareInfo + indexed `get_device_prop` +
  zero-byte workspace. Correct and fixes a latent multi-GPU bug, but measured at ~0.7 µs against a 10.9 µs
  submission that is itself hidden behind 150 µs kernels, and decode replays cudagraphs — submit as hygiene,
  not as a decode win, or not at all.
- **MTP re-measurement + the published Quant Map** — `notes/mtp-remeasure-plan.md`, groups 0a/0/1–5.
- **Draft-vocab 32k slice:** re-measure on the fixed stack (det-135/137 are inside the one-step-behind
  window, det-137 is 2 starts), then it is worth publishing.
- ~~Drafted comments for MiaAI single-Spark #23 and #19~~ **POSTED 2026-09-07** (upstream log 64/65). #19 carries
  a public commitment: our c=16 ceiling is provisional until LOW-1 (cudagraph capture widths) is tested, so that
  item is now owed to a thread, not just to us.

### LOW — leads taken from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, run when the box is idle

1. **Cudagraph capture widths under MTP — the one worth doing first of these.** Prod captures
   `[1, 2, 4, 8]`; with MTP n=3 the decode batch is 4×S rows, so S=4→16, S=8→32 and S=16→64 are **not**
   captured and those steps may be running eager. They report a full decode graph at every verify width
   (4→32) is what made their 8-stream column reachable. ⚠️ Verify the premise first — that vLLM's capture
   sizes are the padded token count and that no capture means eager here — before drawing conclusions.
   If it holds, our c=16 "~100 tok/s is a bandwidth ceiling" was measured on an uncaptured decode path,
   and our null cudagraph A/B (det-136) is consistent because it ran at c=1, the one width already covered.
2. **`MAMBA_SSM_CACHE_DTYPE=bfloat16`** — they measure +8.5 % at 8 streams in a matched pair, needles 15/15
   and per-position acceptance unchanged. We have never tested the recurrent-state dtype outside determinism.
3. **Batch the PLE gather's page faults** — our PLE offload takes 16 page faults per token; never attacked.
4. **Cap the draft context** — they report −17 % single-stream step drafting over 65k rather than 248k.

### Hygiene
- The swizzle harness and its 223 MiB CUTLASS tree still live in an old session's `/tmp` scratchpad, the kind
  the 2026-09-03 reboot wiped. Move both to `/opt/llm/runners` before the next reboot.
- `/tmp/claude-1000` is `drwx------ jschmied` and servers run as `uid=llm`: anything a server must read goes
  in `/opt/llm/runners`, world-readable. This silently invalidated four A/B arms on 2026-09-07.

