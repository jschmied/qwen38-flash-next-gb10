# Open work, ranked

## CURRENT — 2026-09-20 (read this first; supersedes every "Live state" block below)

**Prod runs again, on a different footing than the 09-11 block describes.** Entry point is now the
systemd unit `vllm-flashnext.service` → `/opt/llm/serve-flashnext.sh`, venv `vllm-venv-fnmain3`
(`0.28.1rc1.dev524+g5db652225`), PLE offloaded.

**PROD CHANGED 2026-09-22 (finding 210):** checkpoint is now `qwen38-flash-next-mtpfp4` with
**MTP n=3 + `disable_eagle_block_drop` + local-argmax + the 32k draft-vocab slice**, via drop-in
`/etc/systemd/system/vllm-flashnext.service.d/20-mtp-promote.conf`. **−19.4 % per agent turn**
(1.65 vs 2.03–2.05 s/turn, disjoint ranges, 2 starts each). Was: `fp8head`, no speculation,
24.9/24.5/21.6 tok/s at c=1. Revert = `rm` the drop-in + `daemon-reload` + restart. Two host prerequisites are load-bearing and
neither is in vLLM's config — see memory `flashnext-baremetal-prereqs`:

- **64 GiB swap.** The PLE table is 47.7 GiB and the weights alone are 122.9 GiB against a 121.6 GiB
  `MemTotal`; the table must stay *evictable*. Steady state is 49.8 GiB of swap at 5.3 GiB
  `MemAvailable` over 152 samples. 16 GiB is not enough and the failure is an OOM-kill at ~200/206
  shards with only `Failed core proc(s): {}` in the log.
- **`CAP_SYS_PTRACE`, scoped to the unit** (`AmbientCapabilities`), because `PleOffloadWorker` is a
  *sibling* of the GPU worker so `pidfd_getfd` is a cross-process ptrace op. Host stays at
  `ptrace_scope=1`. Rejected: the host-wide sysctl, and `setcap` on the venv's `python3` (a symlink
  to `/usr/bin/python3.12`, so it would cap every Python on the box).

### IN FLIGHT right now

| job | what |
|---|---|
| `30-probe-heterogeneity` | Can #54076's heterogeneous mamba/attention geometry be reached on dev524? `interface.py:931` is a **floor**, so an explicit `--block-size` cannot create it (816 is raised to 1568) and `:957` pads the mamba page to be *exactly equal*. Probing no-spec vs MTP n=3, since hunk 2's own comment names "a drafter or attention group with a smaller block" |
| `35-devanagari-regression` | Does the U+093E → U+094B substitution below still reproduce on the venv we serve? |

Runner + watchdog are up. `20-load-mainstream` is marked `.done` **deliberately** — it is the job
that OOM'd the box on 09-19 and must not re-run.

### RESULTS from the 2026-09-20 run (jobs 30 + 35)

**#54076's heterogeneous geometry is NOT reachable on dev524 — the arm we owe cannot be run as
specified.** Both probe arms equalize the grids: no-spec sets attention block **1568** and pads the
mamba page 0.13 %; MTP n=3 sets **1600** and pads 0.25 %. The block size changes between arms but
stays equal to the mamba spec *within* each, so `cache_config.block_size == MambaSpec.block_size`
always. `interface.py:931` is a floor (an explicit `--block-size 816` is raised to 1568) and `:957`
does the equalization. This is wickist's own 2026-09-09 finding reproduced on a third configuration.
**What to post instead of the requested cell:** the no-op result with the two log lines, which is
reviewer-relevant for a PR whose author reported the same thing from an RTX 3090 TP2 box.

> ### ⚠️ CORRECTED 2026-09-20 12:44 — the claim below is WRONG; the warning is a false alarm
>
> A two-arm counterfactual (arms differing **only** in `kv_cache_utils.py`, each on its own boot,
> journal scoped to that boot's start):
>
> | arm | warnings | KV | rep2 hits | rate | wall cold/rep1/rep2 |
> |---|---|---|---|---|---|
> | unpatched | **2** | 578,901 | 19,200/26,910 | **71.3 %** | 3.66 / 3.45 / 1.19 s |
> | +#55390 | **0** | 547,693 | 19,200/26,919 | **71.3 %** | 3.61 / 3.42 / 1.18 s |
>
> **Prefix-cache reuse is identical in both arms.** The warning says "prefix-cache reuse across
> requests will be disabled"; in the arm where it fires, reuse runs at 71.3 % with a 2.9x wall-time
> win on the repeat. So on Qwen4Exp/GB10 the warning is **false**, the defect it names does not
> manifest, and #55390 correctly suppresses the annotation while buying nothing measurable. KV
> differs by 5 % but run-to-run spread across five boots was 547k-579k, so that is not attributable.
> **#55390 was reverted from prod** (venv back to the audited 15-file overlay set): an unmerged
> upstream patch with no measured benefit is pure maintenance cost.
>
> Two of my own measurement errors produced the original claim, both now in memory:
> `journal-window-spans-boots` (a `--since "-30 min"` window counted the *pre-patch* boot and I
> reported "#55390 does not fix it" when it had silenced the warning on every post-patch boot) and
> the three-pass rule in `prefix-cache-hit-measurement-trap` (a cold -> one-re-ask harness reads 0
> because align mode does not serve a hit until the **second** repetition -- 0/18,949 then
> 16,000/18,949).
>
> **Still true and still worth reporting:** the warning fires on a configuration where reuse works,
> which is a false alarm worth telling #54360 / #55390 about. **Prod keeps MTP.**

**~~The probe found something worth more than what it was looking for: with MTP enabled, prefix
caching is disabled outright on this build.~~** `kv_cache_utils.py:2160`:

> Speculative decoding (method=mtp) is enabled but no KV cache group could be identified as the
> draft model's, so every group -- including Mamba groups [0, 1, 2, 3] -- will be treated as a draft
> group. A Mamba group cannot satisfy the widened lookup window that implies, so **prefix-cache reuse
> across requests will be disabled** and any external KV offload tier will store without ever serving
> a hit.

KV also drops 869,444 → 571,099 tokens. **Prod runs MTP**, and memory `prefix-cache-works-agent-loop`
measured ~51 % of prompt tokens served from cache and TTFT −40 % after turn 1 — that is what this
costs. This is the engine confirming nickchan0412's root cause on
[#54360](https://github.com/vllm-project/vllm/issues/54360), and the fix in flight is
[#55390](https://github.com/vllm-project/vllm/pull/55390) (Navjot10, open, `kv_cache_utils.py`
+43/−15: annotate the drafter-containing attention group so Mamba groups stay unannotated). It is
also exactly what wickist's #54076 scoping commit addresses from the other side.
**Actions: (a) test #55390 here — we have the reproducing config and the engine's own warning;
(b) decide whether prod keeps MTP until it lands.**

**U+093E → U+094B does NOT reproduce on the current serve.** 6/6 byte-identical, deterministic,
`fp8head` + MTP n=3. **Caveat that matters:** the original was on the **stock RadixArk** checkpoint,
so fnmain3 is *not* cleared for it — only for the checkpoint we serve. First run of the probe
reported `exact=False` on all 6 reps; that was **my harness bug**, `max_tokens=120` against a reply
needing 125 tokens, i.e. a length-stop truncating the copy. `payloads/devprobe.py` now asserts
`finish_reason == "stop"`. The trap is in our own notes and I wrote it into the probe anyway.

### PENDING — upstream, ranked

1. **vllm#54076 — the prefix-cache arm we owe.** wickist accepted on 09-16; we withdrew on 09-17
   when the checkpoint left the box, and it is back. Cell asked for: prefix-cache hit rate on an
   immediate same-prompt re-ask, patched vs unpatched, EOS-correct, 3 starts, optional third arm with
   the scoping commit. Diff **rebased to dev524** (`.dev524.diff`, 3 hunks, hunk 1 dropped as already
   present, hunk 4 reconciled per wickist's 09-06 note). Read it from
   `vllm:prefix_cache_hits_total` deltas — `cached_tokens` is inert.
   **UNBLOCKED 2026-09-22 (finding 208).** Job 30's "geometry not reachable" verdict is WITHDRAWN and
   the public claim retracted on the thread. MaCoredroid's hidden-state-extraction config **does**
   produce the split on our own dev524 build (attention 800 vs hidden-state cache layer 200). So the
   cell they originally asked for — prefix-cache hit rate patched vs unpatched — is **runnable after
   all**, on `qwen38-27b-fp8` with that config rather than on Flash-Next. That is the live next step
   here. Caveat: our build is 777 commits behind main, so any number is "as of 2026-09-08".
2. ~~**vllm#56757**~~ — **CLOSED upstream** (checked 2026-09-22). No action.
3. **vllm#55122 — the tie census is now ANSWERED, by someone else.** `Bizuayeu` posted measured
   evidence 2026-09-22 on 2x GB10 (GLM-5.3-Flash NVFP4, TP=2): ties at the 512th rank do occur in
   real requests — one row had **540 pools, 513 at the 512th value**, and the two pools the kernel
   swapped had bit-identical scores. Their mechanism explains why *our* census found none: the
   indexer logits are a reduction of **FP8** q/k, while we sampled a float32 grid at 16k over two
   prompts. Our 0-ties result is superseded in method, not merely in scope.
   **What is still missing is the perf number** — `MaCoredroid` stated they measured none, so the
   21-28 % claim is unvalidated by anyone. That arm stays **structurally blocked** (main cannot serve
   this model; re-confirmed against the `1ea7c63f4` wheel in finding 209). **A comment adds nothing
   here** — do not post agreement; post a measurement or nothing.
4. ~~**vllm#55430**~~ — **CLOSED upstream** (checked 2026-09-22). The 1.45x union figure is no longer
   owed to anyone; `tools/main/qsa_union_patch.py` stays unapplied and the path gated off
   (`VLLM_QSA_UNION=0`).

### The mainstream move is blocked — RE-CONFIRMED 2026-09-22 against current main

Finding 209 audited the `1ea7c63f4` nightly (= upstream/main HEAD) by extracting the wheel: **it
retires nothing** of our 17-file overlay — the 5 `ple_offload` files are absent from main entirely,
the rest are 0-30 % incidental overlap. #57946, #55122 and #57512 are all still unmerged. So the
upgrade calculus is settled: stay on dev524, and stop re-asking whether main has caught up.

### The mainstream move is blocked, and not for the reason we assumed

PLE offload *is* upstream — but as **#54371 (merged 09-09)**, which supersedes the still-open #53899
and offers exactly two backends, `Qwen4ExpPLEDeviceEmbedding` and `Qwen4ExpPLEPinnedHostEmbedding`.
`allocate_embedding_weight` is unconditionally `pin_memory=True` for the complete table, and pinned
pages cannot be evicted, so on unified memory neither fits. **Do not test it live** — that is the
pinned-PLE condition that has hard-reset this box twice. `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` is not
an escape (it gates a different subsystem). Four open efforts exist and none makes the
unified-memory argument: #54129, #54070, #56757, #57497.

### Closed today — do not re-open

- **Swizzle.** #55180 is merged *and in the serving build* (merge `4df80187` is an ancestor of our
  base, `ahead_by=62 behind_by=0`), and it matters here: 157 of 205 quantized layers are `FP8_PB_WO`.
  #55661 was closed by us on a sound 154-cell structural argument, **not** on the void null recorded
  as finding 147. The promised boundary sweep cannot bite this checkpoint: no layer is within 4 % of
  the 25,165,824-byte L2 boundary. If ever re-opened the question is **N, not M**.
- **`patches/`** archived off-box (`patches-superseded-2026-09-20.tar.gz`, sha256 `e3e754ad88c8…`,
  on `10.0.0.70:/mnt/bulk/gb10/qwen38-archive/`, in `/mnt/bulk/INDEX.md`). Every file in it targeted
  `qwen3_8_flash_next/`, a path dev524 does not have. The live set is `tools/main/`: 15 modified
  files, 8 of them the #53899 port. `patches/MANIFEST.md` stays as the record.
- **vllm#54426** — not ours (Nanetnounou's), fixed upstream by #55557 (merged 09-16), already
  answered in-thread by gaby. Nothing to post; we cannot even confirm it, dev524 predates the fix.


Rewritten 2026-08-31, then appended to per working day. **The sections are chronological, so the
oldest ranking sits at the top — read "Live state" first and treat everything above the 09-06 line
as archaeology unless it is cross-referenced from here.**

## UPSTREAM NEWS 2026-09-22 — three open PRs that change our constraints

**1. vllm#56273 — packed NVFP4 PLE embeddings. DOWNGRADED, probably not worth it for us.**

> **Corrected 2026-09-22 (user):** it buys residency by spending KV. We have already measured that
> the thing it removes is **free**: the PLE offload gather is *completely hidden* — 0.00 ms gap
> before the first PLE kernel ([[decode-c1-idle-piecewise]]) — and *"PLE offload is NOT the
> bottleneck; swap cost per token FALLS"* with concurrency ([[flashnext-concurrency-scaling]]).
> So resident PLE costs **26.8 GiB** of our 869,444-token KV to remove a cost that measures zero,
> and it hurts exactly the axis (concurrency) where offload already wins. The secondary prize —
> retiring the overlay and unblocking mainstream — is real but must be priced against that KV loss,
> not assumed. **Do not chase this without measuring the KV/concurrency trade first.**
*"Enable the `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` checkpoint to run on 1x DGX Spark
**without CPU or disk PLE offloading**. Its packed NVFP4 ngram table occupies **26.822 GiB** and
remains resident."* Validated on one GB10, 4,096-token context, 2 GiB KV, MTP off; load reported
97.47 GiB. **Our 17-file overlay IS the #53899 PLE-offload backport** (finding 209) — if the table
fits resident at 26.8 GiB instead of 47.7, the offload worker, the 64 GiB swap prerequisite and the
`CAP_SYS_PTRACE` unit hack may all become unnecessary, and with them the blocker that keeps us off
mainstream. That checkpoint was already in our HF survey (`local-inference-lab/…-NVFP4`, 23,673 dl).
**Highest-value item on this list.** Open, +431/-28, updated 09-16.

**2. vllm#54912 — THIS IS OUR OWN PR, not news.** Issue **#54552** is ours (2026-08-31); `bojiang3`
agreed with the widening 09-02 and asked for a PR; we opened #54912 the same day and commented
09-11 with det-204 evidence. **Zero human participants in 20 days.** I re-surfaced it as "news" —
check authorship before writing something up as a field finding.
**The band does not need the merge:** we patched the ring locally once already (det-204, cleared on
all 12 QSA layers), so MTP depths 5..8 are **runnable here today**.
It quotes our exact failure verbatim: `QSA ring capacity 12 must divide the attention block size
848`. With `compress_ratio = 4`, depths **5..8 and 13..16** need a 12- (20-) row ring that neither
the power-of-two nor the hybrid 848/1616 sizes have. `qsa_ring_capacity()` returns the smallest
whole-group ring `>= span` that divides the block size, logs when it widens, and errors with numbers
when impossible; previously legal depths keep their ring, so it is a no-op for existing configs.
Our own note calls the untried depth band *"the most concrete untested speed lever we have"*
(`## The MTP depth sweep was cut short by a misread constraint`). **Caveat:** finding 155 measured
k=4 at **−3.4 %** at c=1, so depth alone is not obviously a win — pair any sweep with the dynamic
stopping idea from the EXL3 field read. Open, +98/-11, updated 09-15.

**3. vllm issue #46307 — `gpu_memory_utilization` is not respected on GB10 unified memory; the host
wedges.** *"With a conservative `gpu_memory_utilization=0.70` … available memory to ~7 GiB during
`profile_run` … the entire host becomes unresponsive (SSH dies; requires a hard power-cycle)."*
**Stale — a comment will not move it (user, 2026-09-22); only a PR would.** Do not post evidence
there. **This is our phenomenon, three times over:** the two hard resets behind
[[vllm-load-needs-fit-check]] and [[pinned-sweep-crashed-box]], and the shapebench OOM today where
`MemAvailable` 5.4 GiB was not GPU-allocatable beside a `util=0.90` prod. Only **1 comment**. We have
more evidence on this than the reporter — a candidate post, but note vLLM's policy and get a go.

Also seen, lower priority: #58114 (reduce PLE metadata construction overhead), #57105 (QSA indexer
logits workspace fragmentation), #57039 (GDN: avoid two per-layer copies in FlashInfer prefill),
#56742 (Qwen4Exp MTP buffer placement / config parsing), #56322 (sequence parallelism),
#46329 + #55976 (NVFP4 KV on sm12x), issue #57608 (host-context hook between draft proposal and
verification — the plumbing that *dynamic stopping* would need).

## QUEUED — shapebench the hyper-connection shapes (needs prod DOWN)

`tools/shapebench.py` is the check finding 213 calls for: does FP8 `_scaled_mm` actually beat cuBLAS
BF16 at `(10240, 320)` / `(336, 10240)` at M=1..8, with a control shape FP8 is known to help. Two
minutes of GPU, **but it cannot run beside a live prod.**

**Attempted 2026-09-22, OOM'd.** `MemAvailable` read 5.4 GiB and the tool's peak is ~1.5 GB, so I
ran it — and it failed on the first `torch.randn`. **`MemAvailable` is not a proxy for
GPU-allocatable memory when vLLM holds a `gpu_memory_utilization=0.90` reservation**: on unified
memory the host sees free pages a second CUDA context cannot obtain. The failure was clean (Python
exception, MemAvailable 5.4 → 5.1, no reset), but the guard was measuring the wrong quantity.

**Do not shrink the working set to make it fit.** The ~300 MB rotation exists to defeat L2; the
tool's own docstring notes that anything faster than roofline is timing cache. A smaller run would
produce a confident number that is false.

**Run it in the next prod-down window**, before any quantization work on these shapes. Expectation
from 213: FP8 loses here, because the shape is padded specifically for cuBLAS BF16 heuristics.

## FIELD READ 2026-09-22 — the EXL3 GB10 recipe: what transfers and what does not

Source is an **EXL3 (ExLlamaV3)** recipe, a different engine, so every percentage in it is relative
to *its* baseline. Our standing pattern applies: three MiaAI-Lab numbers (+6.8 % bf16 SSM state,
+11.4 % MTP k=4, +5-6.6 % CPU idle states) all failed to transfer to us. Treat the list as levers,
not results ([[check-field-before-expensive-steps]]).

### Already ours — corroboration, no action

| their item | our evidence |
|---|---|
| smaller draft-head projection, full head kept for verification | **our draft-vocab slice**: 32k of 248,320 rows, det-135, +6.4-6.8 % at c=1. Shipped, and in the config promoted 2026-09-22. |
| INT8 activation GEMV **off** — FP16 GEMV was faster | our own law, `block-size-is-not-a-kernel-limit.md:98`: removing bytes is not sufficient, you must also land on a kernel at least as good. cuBLAS skinny BF16 GEMV beats the Triton blockwise path at `(10240, 320)`, M=1. **Finding 206 (DeepGEMM 8 % slower) is the third instance of this same law.** The most credible item in their list, because it is counterintuitive and we hit it independently. |

### TRANSFERS — worth a run, in order

1. **INT8 / MXFP8 on the BF16 leftovers** (their "INT8 mixer weights, 7-13 %"). Maps onto our single
   largest untouched decode lever: **16.5 % of kernel time** in BF16 GEMVs — shared expert, hc
   low-rank, router, MTP dense (det-136 ranking). **Constraint already known:** those shapes are
   `K=320`, *not* 128-divisible, so blockwise FP8 **cannot express them**
   (`single-stream-limit.md:340`); needs a group size dividing 320 — MXFP8 (32) or INT8 per-channel.
   Their 7-13 % is a plausible magnitude for exactly this work. Also needs the **four hardcoded
   `quant_config=None` opt-outs** in the model removed — quantizing is a two-part change.
2. **Dynamic stopping / adaptive draft length** (their "five-token MTP with dynamic stopping").
   The *depth* does not transfer: finding 155 has k=4 at **-3.4 %** at c=1, and k=5 is illegal in our
   stack — QSA's ring needs `block_size % capacity == 0`, and at block 1600 n=5 gives capacity 12
   with `1600 % 12 = 4`. EXL3 has no QSA ring. **The stopping rule is the transferable idea**, and it
   is already queued ([[dflash-27b-next-levers]]: nspec 8 short / 12+ long). Untested.

### DOES NOT TRANSFER / lower prior

- **Wider fused-MoE decode tiles** — tension with our own measurement: MoE GEMM1 is already at the
  **DRAM floor** ([[moe-grouped-gemm-dram-floor]]); our fix there is epilogue fusion, not tiling. A
  wider tile cannot beat a bandwidth bound. Would need a shape that is *not* floor-bound to pay.
- **Fewer host synchronizations** — our c=1 decode step is **GPU-bound**; the 46 % "idle" was
  profiler overhead (det-134/136). Low prior.
- **CPU launch threads pinned to big cores** — we measured the sibling lever (CPU idle states off)
  at **0.7-0.9 %** against the field's claimed 5-6.6 % (finding 156). Same class, did not transfer.
- **8-bit KV cache** — already wanted, not new: #55557 (fp8 KV x1.72), **blocked behind #53899**.

## QUEUED 2026-09-10 — quantize the MTP module's own body (memory, not bandwidth)

**Not the MTP head — that question is closed.** MTP has no head of its own; it shares the main
`lm_head`, and an NVFP4 head is not viable (FP8 reconstructs at 2.642 %, NVFP4-max at 9.491 %, a
6.85 pp gap against Local-Hessian's demonstrated 1.35 pp). Head stays FP8. See
`quantizing-lm-head.md`.

**What is open is the MTP module's own weights:** 4.69 GiB of BF16 experts + 0.17 GiB dense, all of
it in RadixArk's `exclude_modules` as `mtp.*`.

**The case is resident memory, not bandwidth.** Its experts are MoE, so only ~0.092 GiB of that
4.69 is read per draft token — quantizing saves ~0.069 GiB/draft-token, an order of magnitude less
than `lm_head`'s lever. What it does buy is **~4.1 GiB freed resident**, and `fp8-mixed-checkpoint.md`
showed freed resident converting straight into KV cache. That helps exactly where MTP is weakest:
c≥8, where speculation measured a net loss (−6 % at c=8, −10 % at c=16 on a BF16 head).

**The blocker is identified and already fixed upstream.** Our venv has `_remap_ignored_layers` in
`qwen4_exp/nvidia/mtp.py` but **not `_remap_quantized_layers`** — which is what
[vllm#55513](https://github.com/vllm-project/vllm/pull/55513) adds (**MERGED 2026-09-08**, "Fix block
FP8 MTP in ModelOpt mixed checkpoints"). Without it, MIXED_PRECISION reads `quantized_layers` with
main-model indices while MTP renumbers its layers, and loading fails with
`mtp.layers.N.mlp.experts has no parameter 'w2_weight_scale_inv'` — the failure two other DGX Spark
users already hit. Our venv is `0.28.1rc1.dev401+g8340fe1bb`, older than the merge.

We drafted this same fix ourselves and deleted the draft once #55513 landed — correctly. Do not
re-derive it (memory `search-open-prs-before-fixing`).

**Steps:** (1) port #55513 via the `venv-overlay` skill, or bump the venv; (2) quantize the MTP
experts with **plain max** — no capture needed, since stage 1 hooks only the main model's MoE blocks;
(3) serve and measure **acceptance length** and resident memory, not just tok/s.

**The cell that decides it:** mean accepted length. The `lm_head` result (2.21 vs 2.15, unchanged)
covers a layer the drafter *reads*; its own body determines what it *proposes*, which is a different
question. If acceptance holds, the freed KV is a clear win. If it drops, the bandwidth saving is far
too small to pay for it.

## THE NIGHT OF 2026-09-08/09 — HISTORICAL (was 'read this first'; see CURRENT, 2026-09-20)

**Goal, reset by the user: make agent turns faster.** Eight runs on the box, each with the cell that
must differ named before launch. Findings 151–158 in `prefill-investigation.md`, det-184 in
`determinism-investigation.md`, raw data in `notes/data/`.

### The headline, and it is not a speed result

**`tcorrupt` (det-185): mmastrac's tool-call-corruption repro from vllm#54521 gives 40 distinct
completions from 40 identical greedy requests on stock, and 1 with our four fixes.** 49,902-token
agent transcript, TP=1, divergence starting at token 0 or 1. Their own 4× GB10 TP=4 run gave 5
distinct in 40 with corruption at token 23–31 — ours is categorically worse, on a different model.
A corrupted tool *name* makes the parser emit zero deltas and the request finishes with no content
and no tool calls, i.e. **a whole agent turn lost**, which costs more than the best speed lever
measured tonight (−9.6 %) buys. Draft reply written (`notes/upstream/comment-54521-tcorrupt.md`),
**not posted — needs your go**, and it is the strongest evidence we have produced for #55122's set.

### Follow-up run, 2026-09-09 morning (user: "run it")

**det-186 — the concurrent nondeterminism is NOT an MTP defect.** With speculation off entirely, 8
concurrent greedy repeats still perturb **2,503 of 2,504** scored positions starting at position 1,
against **0 of 2,504** sequentially in every arm. MTP amplifies it about 2× (median per-position
spread ratio 2.25; flips 179 → 234; 88.8 % of the MTP-off flips are a subset) but does not cause it.
**And it is deterministic**: the exact flipped-position set is identical across two independent server
starts, so it is a reproducible function of batch composition, not entropy — bisectable, and fixable
by making a reduction batch-independent. Next rung proposed but not started: vary
`--max-num-batched-tokens` (one A/B) before any instrumentation, because `layerhash_patch.py` needs
per-request slicing before it can answer this.

### What needs your decision

1. **`--mamba-ssm-cache-dtype bfloat16` (finding 153) — the only real win of the night, and it has a
   price.** Attention block 1,600 → 832, KV capacity +21–36 %, and **total agent-turn TTFT −9.6 %**
   over 24 paired turns. But it is a *tail* lever: the median turn gets slightly worse, the expensive
   turns get much better. And it changes **127 of 2,504 modal top-1 predictions** — the same order as
   the four determinism defects we spent a week removing. Adoption is one `FN_SSM_DTYPE` line.
   **My recommendation: do not ship it on this evidence; run a task-level eval first.** A logprob
   count cannot tell you whether a 5 % shift in predictions costs anything real, and the field
   shipped it on "needles 15/15", which could never have seen it.
2. ~~**Prod is still on `vllm-venv-fnmain`**~~ **DONE 2026-09-11 (det-206): prod is cut over to
   `vllm-venv-fnmain3`.**

### ~~PROD STATE as of 2026-09-11~~ — SUPERSEDED 2026-09-20

That table named `/opt/llm/serve-fnmain.sh` as the entry point and "one local patch". Both are now
wrong: prod is the `vllm-flashnext.service` unit → `/opt/llm/serve-flashnext.sh`, and the venv
carries **15** modified files (8 of them the #53899 PLE-offload port), verified by diffing against
`vllm-venv-fnmain3-vllm-pkg-pristine-dev524.tgz` rather than by reading patch files. See **CURRENT**
at the top. Still true from the old table: FlashInfer `0.6.18.post1` (deliberate mismatch against
vLLM's `==0.6.18` pin, det-206), the #55715 GDN backport, and that 27B prod is a separate unit.

### What was measured and rejected — no action needed

- **MTP k=4** (155): −3.4 % at c=1, no better at c=8. It buys draft length and loses acceptance rate;
  on one prompt the rate collapses 53.2 → 35.7 %. Not a prod change.
- **FLA fused kkt+solve** (154): −1.4 % cold TTFT at 8k, −1.1 % at 30k, **null** on warm agent turns.
  Closes finding 143 as a correct kernel win that does not move agent turns. The PR draft still
  stands on kernel merit and needs your go.
- **CPU idle states off** (156): 0.7–0.9 %, not the field's 5–6.6 %. Not worth a host-wide setting on
  a PD-limited box.
- **Async scheduling** (158): bit-identical output at c=1 (0/2,504). The caution is dead, default stays.

### Three field numbers did not transfer, and that is the pattern of the night

MiaAI-Lab's +6.8 % decode from bf16 SSM state, their +11.4 % from MTP k=4, and their +5–6.6 % from
disabling C-states all failed to reproduce here — while the *mechanisms* they described were all real
(the block did halve, the k-curve does exist, the idle-exit latency is 42 µs). Their measurements are
not wrong; they are on a dual-Spark TP=2 pair with a different checkpoint. **Treat a field number as a
hypothesis about a mechanism, never as an expected effect size.**

### Four things we learned about our own measurements

- **Acceptance is a speed statistic, not a quality signal** (158): async and noasync differ by 1.1 pp
  of acceptance with **bit-identical output**. And acceptance is chaotic at the ulp level (154): a
  one-ulp kernel change flipped it ±10 pp in both directions. Acceptance figures are not comparable
  across builds that differ numerically at all.
- **Rank warm-turn levers on the paired per-turn total, never the median** (153). Finding 142 called
  the bf16 knob flat because it looked at the median; the mean and the total say −9.6 %.
- **A mechanism check must test a captured string, not a pipeline's exit status** (157). `ishare`
  voided six arms on `grep … | head -1 || echo NOT-FOUND` — `head` always succeeds. Fifth void run,
  first one caused by the gate rather than the knob. Rule added to `method.md`; re-queued as `ishare2`.
- **A launcher caution that the launcher does not enforce is worse than no caution** (152/158). Ours
  said never to combine MTP with async scheduling; the library default enabled it anyway, so we ran
  the forbidden configuration in every measurement for weeks while believing we were safe.

### Also done

Field + open-issue sweep (`the-field.md`): three levers the field had measured and we had not, one
(PLE `posix_fadvise`) struck as inapplicable, and two upstream issues identified as the same
mamba/attention page coupling from different sides. **Finding 151:** our draft-vocabulary coverage
curve saturates at a tenth of the field's size — the whole observed vocabulary is 48,476 ids and
held-out agent *output* has 2,048 — so their 65k does not exist here and their own crossover rule
points at ~4k, not our shipped 32k. **det-184:** the four determinism fixes are jointly necessary and
individually near-worthless (280–334 of ~333 disagreeing positions survive any one of them; all four
give 0), and they move the *modal* answer at 110/2,504 positions, not just the variance. README's
fix list reconciled against what the venv actually carries: five defects, not three or four.

### Housekeeping done at the end

**Disk: 30 → 35 GB free.** 71 per-arm compile/autotune cache roots from closed runs (`mr-`, `cgs-`,
`pd-`, `cg-`, `i5-`) removed once the box was idle — 5.4 GB, all from runs whose findings are written
and pushed. Tonight's roots are kept. `/opt/llm/.cache-fnmain2` is 10 GB → 4.7 GB, 34 roots left.
Per-arm cache roots are required by the compile-cache-key trap, so this directory grows by ~90 MB per
arm and needs a purge after every campaign.

<!-- removed 2026-09-20: tcorrupt / ishare2 runners no longer exist. -->

### Drafted, not posted — all need your go

`comment-54521-zc502-isolation.md` (the owed collector validation, now carrying det-184's isolation
table), `comment-miaai-19-cudagraph-widths.md` (why our capture-size arms were null and where the
effect actually lives), and the standing `pr-fla-fused-kkt-solve.md`.

## WATCH — PixelML DFlash drafter for Flash-Next NVFP4 (added 2026-09-09)

https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash — uploaded **today 19:05**, 1 like,
0 downloads. **Weights are not up yet: the repo currently contains only `README.md`.** So this is a
watch item, not a download.

**What it is:** a DeepSpec **DFlash block drafter** (5 draft layers, block size 7, one parallel pass) for
`nvidia/Qwen3.8-Flash-Next-NVFP4` — ships draft layers plus the fusion projection only, not a full model.
Their base is **NVIDIA's** build; ours is RadixArk, so transfer is not automatic.

**Why it is interesting to us:** DFlash beat MTP decisively on our 27B (34–40 t/s vs MTP's 23.6). A
DFlash drafter for Flash-Next would be the same swap on the model we actually serve.

**Why it is LOW priority anyway — their own README disqualifies it for our workload:**
> "this is a maths drafter, a code wash, and it makes chat slower at every block size we can serve.
> It beats the target's own tuned speculative head by 3.87 % in aggregate, in eager mode"

We run agent/code traffic, which is the "code wash" case, and we serve with cudagraphs, not **eager**.

**Their protocol is unusually close to ours** — 2× DGX Spark (GB10), vLLM, TP2 + expert parallel, c=1,
one frozen 100-prompt fixture (33 code / 33 math / 34 chat). Two things worth taking from it regardless
of whether we ever run the drafter:

1. **They independently corroborate our finding 155.** Against MTP k=4 they measure k=3 at **−0.60 %,
   95 % CI [−1.47, +0.25]** — i.e. k=3 and k=4 indistinguishable, k=6 at −6.72 %, k=1 at −17.97 %. Our
   finding 155 had k=4 **worse** than our shipped k=3 by 3.4 %. Same conclusion, different hardware
   topology: **do not move off k=3.**
2. **Their blog is a methodology read**: *"Our 25 % inference speedup became 3.9 % after we fixed the
   benchmark twice"* — https://github.com/PixelML/deepspec-qwen38-flash-next/blob/main/blog/README.md

**Action when weights appear:** check whether the drafter is tied to NVIDIA's checkpoint or transfers to
RadixArk, and measure on **agent** traffic with cudagraphs on — the two axes their own numbers do not
cover. Do not adopt on the strength of an eager-mode aggregate.

## HISTORICAL — live state 2026-09-08 night (goal: agent turn time)

**The goal was reset by the user tonight: make agent turns faster on this model. The top-k /
determinism work was a divert that the errors we found made necessary; it is finished and posted.**
An agent turn is TTFT + tokens/rate with TTFT 53–69 % of it and the median turn emitting ~130–242
tokens, so prefill and warm-turn recompute are where the time is.

**Chained on the box tonight** (systemd units, each waiting on the previous by name; watchdogs set
just after each expected finish):

| # | unit | question | named cell that must differ |
| --- | --- | --- | --- |
| 1 | `isolate5` | which of the four determinism fixes carries end-to-end reproducibility | `none` non-zero AND `all4` ~0, or void |
| 2 | `ssm2` | **DONE 23:02 — finding 153.** Block 1,600 → **832** as predicted, KV +21–36 %, intercept −51 %. Real agent turns: **total −9.6 %** over 24 paired turns (18.84 → 17.04 s, faster on 15/24) but the **median turn is slightly worse** — it is a tail lever. Costs **127/2,504 modal top-1 changes**. No decode win; vllm#55533 does not reproduce. Prod adoption = one `FN_SSM_DTYPE` line, **user's call, pair it with a task eval** | ✅ the block-size log line differed |
| 3 | `pstack` | **DONE 00:36 — finding 154.** Cold TTFT **−1.4 % at 8k, −1.1 % at 30k** (ranges do not overlap), **null** on warm agent turns. Its apparent 5–10 % decode win is a **1-ulp numerics change flipping MTP acceptance ±10 pp** — expectation zero. Closes finding 143 as a correct kernel win that does not move agent turns; the PR still stands on kernel merit | ✅ marker present in `fla`, absent in `base` |
| 4 | `mtp42` | **DONE 02:19 — finding 155.** k=4 is **−3.4 %** at c=1 (buys draft length, loses acceptance rate; one prompt collapses 53.2 → 35.7 %) and no better at c=8. **Not a prod change**; the field's +11.4 % does not transfer. **vllm#55533 does not reproduce** — scheduler reaches width 8 in every arm and no-spec is 19.9 tok/s against n3's 23.4–25.3. Its `1+k` mechanism *is* visible in the block size (1,568 / 1,600 / 1,616 at k=0/3/4) | ✅ n3 vs n4 differed |
| 5 | `ishare3` | **DONE 07:44 — finding 159.** Index sharing is **output-preserving by construction** (0/2,504 mismatches; it touches only the drafter, and greedy MTP stores the target's argmax) and worth **−0.5 % at c=1, −0.33 % on agent turns**. Leave `FN_ISHARE` off. Fourth field number that did not transfer | ✅ gate clean at the third attempt |
| 5a | `ishare`/`ishare2` | **VOID on gate bugs of mine, not on the flag — finding 157.** `grep \| head \|\| echo` takes head's exit status, so the marker was empty and the gate declared the base arm contaminated. The flag *had* engaged. Two arms that ran suggest no gain (AL 2.64/2.17/2.60 vs 2.64/2.15/2.63). **Re-queued as `ishare2`** | ❌ gate bug; rule added to method.md |

Runs 2–5 all came out of a field/issue sweep done the same evening (`the-field.md`, 2026-09-08) —
three of them are levers the field has measured and we had not, and one (`pstack`) is a finding of
ours that had never been taken to the server.

**Done tonight, do not redo:** field + open-issue sweep (`the-field.md`); finding 151 (our
draft-vocabulary coverage curve — the whole observed vocabulary is 48,476 ids, so the field's 65k
does not exist here and their own crossover rule points at ~4k, not our 32k); finding 142 corrected
twice (the bf16 route exists; fp8 KV *doubles* the attention block, which is the entire difference
between MiaAI's 3,200 and our 1,600); the memory `warm-turn-block-granularity` corrected (vllm#54458
is the open issue for the allocator change and we are its second commenter — the earlier "no open
issue exists" was a search failure).

**Owed, and now sharpened by a third party (2026-09-09 01:11):** rybruscoe on vllm#54521 proposes the
discriminator our own PR #55122 needs — log the selected index set and the k-th indexer score for two
byte-identical requests above the budget, and check whether the differing indices sit inside a near-tie
band. If they do, the *scores* differ run to run (reduction order rounding) and the selection kernel is
only where it becomes visible; if the differing indices have clearly separated scores, it is a genuine
ordering bug in the top-k. They report the same shape from reward scoring: eight valid BF16 reduction
orders flipped 52 verdicts, all near the threshold, and exact accumulation flipped none.
**det-184 is already indirect evidence for their reading** — if top-k ordering were the whole story,
`qsadet` alone would have removed far more than 0.9 % of the divergence instead of 3 positions out of 333.
The instrument exists: the `tiecensus` runner (fixed after its chown failure, never re-launched) measures
exactly `ambiguous iff n_gt < k < n_gt + n_eq` on real indexer scores. **Top candidate for the next idle
slot**; it decides whether #55122's stated premise is right, which matters before anyone merges it.

**Runs 6–8 (added during the night):** `cstates` **DONE 03:15 — finding 156**, CPU idle states cost **0.7–0.9 %** of a decode step, not the field's 5–6.6 %; not worth a host-wide setting. `asched` (running) — was our "never combine MTP with async scheduling" caution harmless, given we have been doing it for weeks. `tcorrupt` — mmastrac's tool-call-corruption repro from vllm#54521, stock vs our four fixes. `ishare2` — the re-run.

**Prepared, not yet run:** draft-vocabulary size sweep at 4k/8k/16k vs 32k (files built, `dv_patch.py`
is the hook; 32k must be re-measured because the rebuild overwrote det-135's files). CPU C-states off
(`cpupower idle-set -D 0`) — MiaAI measured +5–6.6 % by live ablation and it is the only host-side
lever anyone has found; needs one server and two interleaved ablations, no restart.

**Env note:** `/opt/llm/serve-fnmain.sh` gained two opt-in knobs tonight, both default-off:
`FN_SSM_DTYPE` (→ `--mamba-ssm-cache-dtype`) and `FN_ISHARE` (→ `index_share_for_mtp_iteration` in
the speculative config). Backup at `/opt/llm/serve-fnmain.sh.bak-20260908`.

## HISTORICAL — live state 2026-09-08 (end of day)

**In flight:** nothing on the box; it is idle.

**RESOLVED 2026-09-08 night — the three lists are reconciled**, with the venv checked rather than assumed: the set is
**five** defects. #55375 (PLE state-stride) is merged upstream and byte-identical in `vllm-venv-fnmain2`, so it is not an
overlay; the other four are what `prod_det_overlays.sh` installs and what det-184 measured; the FlashInfer autotune cache
key was missing from every earlier list even though the MoE finalize fix cannot start without it. README fixed. The
superseded audit follows for the record:

**CAUTION — three different "the fixes" lists are in circulation and they disagree (audit 2026-09-08):**
- **PR #55122 body (live, public):** "one of **three** independent defects" — top-k + MoE fused
  finalize (#54945/#54948) + align-mode blocks (#54076/#53798).
- **README:** "**four** fixes" — top-k + MoE finalize + PLE state-stride (#55375) + PLE offload
  semaphore.
- **`prod_det_overlays.sh` (what we actually run):** qsadet + detfin + **cachekey** (FlashInfer
  autotune key) + **plefix** (PLE offload semaphore).
The PR body's list omits the PLE offload semaphore entirely — the fix `isolate4` may show is
load-bearing, and which det-182 confirms is **not upstream in any form**. Reconcile all three before
the next upstream post, and say plainly which set was actually measured together.

**Owed to people who asked us directly:**
- **ZC502, #54521** — their client collector is validated on sm_121 (12/12 runs, det-178) and that
  can be reported. But our det0/det1 case was **void**: stock did not diverge on any sequential
  case, because random-word prompts do not tie at the top-k boundary. **Re-run with real prose near
  `indexer_budget` plus a deliberately tie-heavy case before offering them any determinism numbers.**
- **MiaAI #19** — still owed. Needs a wide-capture-size arm; `cgsize2`, which was supposed to be it,
  was vacuous (det-170's class), and `cgnone2` did not discharge it (det-169).

**Done today, do not redo:** #55122 answered and updated (review conceded on accuracy, default-change
position dropped, two perf commits folded in — det-171/172/173, PR head `7cfd04a39`); #55872 tested
and reported (det-175, fails to init on sm_121); fnmain3 built, hand-ported and **proved working**
(det-177); `fnmain-overlay-dev524.diff` regenerated and committed.

**Blocked:** `zsign`'s "theirs" arm will not build — #55314's kernel needs its own driver
(`bench/topk-union/their/topk_their.cu` + `bindings_their.cpp`), so the ±0 set-defect claim stays
unmeasured and `notes/upstream/comment-55314-signed-zero.md` stays unposted.

**Decision waiting on the user:** cut prod over to fnmain3. `serve-fnmain.sh` still defaults to
`vllm-venv-fnmain`; the switch is one `FN_VENV` line. Anything measured after the cutover is not
comparable to det-169/171/172/173, all taken on dev401.

## Open, in order (ranked 2026-08-31 — item 1 is now closed, see below)

1. **Corroborate or refute vllm#54521's `indexer_budget` model.** They report greedy decoding
   deterministic *below* `indexer_budget` (2048) and non-deterministic above, because QSA switches
   to top-k selection — which would explain our own unexplained non-determinism. **Our first run
   contradicts them:** divergence at 582 and 1,142 prompt tokens, both well below the budget. But we
   run MTP k=2 and their repro does not, so the decisive cell is MTP-off below the budget. In
   flight. Either outcome is worth reporting; the issue has no comments.
   → **CLOSED 2026-09-08. The `indexer_budget` model is refuted**, and not only by us: davidcanar
   reproduced on gfx1151 *without* QSA at all, sudodrew on SM120, mmastrac on GLM-5.3-Flash at TP=4.
   The issue now has 39 comments and five separate defects. Our own det-178 adds that stock can be
   perfectly self-consistent at 5,960 tokens — far *above* the budget — so length is not the
   discriminator; ties at the top-k boundary are. The issue title is now too narrow for what the
   thread contains and arguably mis-routes triage.
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

## QUEUED: two concrete jobs from 2026-09-12

1. **Fix the det-kernel shared-memory budget (det-222 addendum).** Hoist `cudaFuncGetAttributes`
   above the `chunk_size` computation in `/opt/llm/kernel-det/topk_det.cu` and subtract
   `fa.sharedSizeBytes` from `effective_max_smem`; make the residual case fall back to the stock
   kernel instead of `STD_TORCH_CHECK`. Rebuild via `build_det.py`, validate with the **100k arm of
   `filedrift.py`**, which is the run that exposed it. Do NOT clamp `smem_size` at line 116 — it is
   `chunk_size` in bytes and the kernel indexes it.

2. **Test vllm#55872 on this box — publicly promised 2026-09-12.** Pure Python, and it touches our
   exact path (`qwen4_exp/nvidia/indexer_qsa.py`, `ops/qsa_indexer.py`), so: clone a venv, apply the
   PR's diff, serve, and check the opt-in deterministic backend selects and runs on sm_121. We told
   @LopezCastroRoberto we would report back.

## REGRESSION HUNT — U+093E → U+094B (raised 2026-09-11, STILL OPEN, re-testing 2026-09-20)

**The bug.** Stock RadixArk checkpoint, prompt "copy this exactly":
`उपयोगकर्ता को पहले लॉग इन करना होगा।` → fnmain2 returns it **exactly** (6/6 byte-identical);
fnmain3 returns `उपयोगकर्तो को पहले लॉग इन करनो होगा।` (6/6 byte-identical). Every open-syllable
**U+093E (AA matra) → U+094B (O matra)**. A substitution, not duplication (det-211b). Deterministic
on both sides, so **one request settles an arm** — `/opt/llm/runners/devbisect.py`, ~13 min per
hypothesis.

**Three candidate causes**, because fnmain3 changed all of them at once:
1. the **#55715 GDN prefill kernel** — det-212, running
2. **FlashInfer 0.6.17 → 0.6.18.post1** — note det-208: identical sm120 cubins, so it would have to
   be the Python/JIT path
3. **~123 vLLM commits**, dev401 → dev524 — binary search over nightlies, ~7 rungs, last resort

**Inserted before 2 and 3: MTP off.** A deterministic single-token substitution is what a
speculative-decoding acceptance bug looks like, MTP 3 is on in every arm measured so far, and the
rung costs one server start with no venv work.

**Prod is on fnmain3 and carries this.** The revert is one line (`FN_VENV` →
`vllm-venv-fnmain2`) and is the user's call.

> **2026-09-20:** never resolved, and it is the highest-priority open correctness item — we are
> serving fnmain3. Job `35-devanagari-regression` re-tests it. One caveat on whatever it returns:
> the original was measured on the **stock RadixArk** checkpoint and we now serve the derived
> `fp8head` one, so a clean result does **not** clear fnmain3 for the stock checkpoint.

<!-- removed 2026-09-20: runners fx-gdnab-driver / fx-thaidet-driver are long gone (box repurposed to DS4.1 and back). The U+094B question it raised is preserved in the REGRESSION HUNT section. -->

## Work queue as of 2026-09-07 (user: "higher prio has work on our own findings and PRs, run when idle")

### HIGH — our own findings and PRs

- ~~**OWED: PR for our own issue #54552**~~ **WRONG — [PR #54912](https://github.com/vllm-project/vllm/pull/54912)
  has been open since 2026-09-02.** Not late; `REVIEW_REQUIRED` with no human review in nine days and
  CI blocked by the `pre-run-check` label gate (same as #55122). Nothing owed from us. The PR's
  implementation already bounds the widening at 2x, which today's patch did not.
  **What is genuinely new:** runtime evidence the PR lacks — the widening fired on all 12 QSA layers
  and cleared the assert (det-204). Worth a comment there if we want to move it. Original note:
  The QSA ring assert makes `num_speculative_tokens` 5..8 unreachable. Maintainer agreed with the
  proposed widening on 09-02; we replied "I can open PR of course." **The patch is now written and
  proven at runtime** (det-204: fired on all 12 QSA layers, 12 -> 16, block size 1616, span 9, cleared
  the assert) — `/opt/llm/runners/qsacap_patch.py`, copy in `notes/data/`.
  Still to do: turn it into a proper diff against main, a test, and the vLLM PR format (`git commit -s`,
  AI-assistance disclosure, Purpose / Test Plan / Test Result). **Needs the user's go to open.**
  Note the patch must use *next multiple of `compress_ratio` that divides block_size* (the wording in
  our issue), not merely "smallest divisor" — they coincide at 16 here but diverge in general.

- **TEST A — patch `qsa_cache.py` so ngram can serve (det-203).** We *can* fix this; I wrongly called
  it a maintainer's call. The safety invariant in the source comment is `capacity >= span`
  ("anything narrower lets a rejected draft row overwrite a committed key"); "whole groups" is only a
  *mechanism* for divisibility, and it achieves nothing here because the ring never joins the LCM
  (`CircularBufferSpec.prefix_cacheable` is `False`).
  **Patch:** `capacity = smallest divisor of cache_config.block_size that is >= span`. For n=5 that is
  **16 instead of 12** — safe (16 >= 11) and divides 1616 by construction.
  Via `venv-overlay`: backup `.orig-qsacap`, on/off script, dry-run round trip on copies first
  (rule 7 caught a real bug in the PLE port). **Prove the path at runtime** — log the chosen capacity.
  **Decides:** whether ngram serves at all on this model, which then makes the Quant Map's ngram cells
  (29.0 ms/tok agent claim) testable for the first time. If it serves, check output coherence before
  any timing — a too-small ring corrupts silently, and that is exactly what this assert guards.

- **TEST B — 27B FlashInfer large KV pages (first non-inert smgates hit).**
  `v1/attention/backends/flashinfer.py:422` gates `use_large_pages` on
  `is_device_capability_family(100)`; the 27B meets every other condition (24 Q / 4 KV, GQA 6) and
  **uses `AttentionBackendEnum.FLASHINFER`**, unlike Flash-Next. So on sm_121 it is capped at KV block
  sizes `[16, 32, 64]` and never offered >=128.
  **Cell that must differ:** the advertised `kernel_block_sizes`. Patch the family check to admit 12x,
  confirm >=128 is offered, then A/B TTFT + c=1 decode on the 27B, three starts, **bracketed with
  `bwprobe.py`** (det-201).
  **Precondition CLEARED 2026-09-11 (static, no GPU).** `supports_trtllm_attention()` in dev524
  already admits us: `is_device_capability_family(120)` returns `not is_prefill`, i.e. **XQA decode
  yes, TRTLLM prefill no**. `has_nvidia_artifactory()` is True (satisfied by the installed
  `flashinfer-cubin`, no network needed). 24 % 4 == 0, so `can_use_trtllm_attention(24, 4,
  is_prefill=False)` is True by construction. The gate is now at
  `v1/attention/backends/flashinfer.py:437` and is still family(100)-only while every other
  condition (`num_qo_heads // num_kv_heads == 6 > 1`) holds — same shape as #55715.
  **So this does not close for free; it is a real A/B.** Still unverified: whether larger KV blocks
  help at head_dim 256. Decode-only, so measure **c=1 decode**, not TTFT.

- ~~**VENV BUMP: we have never used the FlashInfer GDN prefill kernel (det-202).**~~ **DONE 2026-09-11** (det-205/206): fnmain3 = dev524 + FlashInfer 0.6.18.post1 + the #55715 backport, prod cut over; the 0.6.17 blocker below was itself refuted in det-208. Our own A/B is det-207.

- ~~original note:~~ #55715 merged
  2026-09-08 enables it on SM12x; our 1,216 logged GDN announcements are all Triton/FLA. The PR
  measures **7.2 % TTFT on a GB10** at ISL 32768 and 3.8–4.5× on the kernel itself. **Needs FlashInfer
  ≥ 0.6.18 (we are on 0.6.17)** plus a venv past 09-08. This is the largest measured item on the
  prefill/TTFT goal and it is someone else's merged work, not ours to build.
  Bundled in the same bump: #55272 (removes torch.compile for this model — bears directly on det-194's
  open cudagraph question), #55170, #54110, #55513.

- **Scan vLLM for sm_12x gates that could be opened (user request 2026-09-11).**
  **Scanner v2, AST-based** — `/opt/llm/runners/smgates.py`, output `notes/data/smgates-v2-dev401.txt`.

  **v1 was regex-based and MISSED vllm#55715** (det-202), the single most valuable gate found all day:
  `_resolve_gdn_prefill_backend()` set a `supports_flashinfer` boolean from an SM-family check and
  returned "triton" when unset, with the constraint documented in a *docstring*. No literal to match.

  v2 walks the AST **per function** instead of per line, so a flag-mediated gate is visible: it flags
  a function that names a specific SM arch (literal, `is_sm90()`-style call, or docstring) and never
  mentions 12x. It then ranks by whether a capability-derived `supports_*`/`use_*` boolean exists (the
  #55715 shape), whether the file is on a path this stack actually executes, and whether the
  enumeration is Blackwell-datacenter-only. **v2 re-finds `_resolve_gdn_prefill_backend()` at rank 6**,
  which is the regression test that matters.

  387 functions enumerate an arch; 71 are 12x-aware; **152 suspect** after dropping tests, third_party
  and other vendors. Top new candidate, score 9:

  | site | gate |
  | --- | --- |
  | `v1/attention/backends/flashinfer.py:422` `get_supported_kernel_block_sizes()` | `use_large_pages` requires `is_device_capability_family(100)`, so sm_121 never advertises KV block sizes ≥128 |

  **Likely inert for us, same as det-180 and the FA4 gate:** we do not use the FlashInfer *attention*
  backend — the model runs its own QSA state backend and the vision tower uses FLASH_ATTN; FlashInfer
  serves our **MoE**, not attention. Worth re-checking for the 27B, whose attention differs.

  The v1 list, for the record:

  | site | gate | note |
  | --- | --- | --- |
  | ~~`v1/attention/backends/fa_utils.py:233`~~ | `capability.major in (10, 11)` | **CLOSED 2026-09-11, inert.** FA4 is never selected on our stack: across every `fnext-*.log` the only attention announcement is `FLASH_ATTN for vit attention`, and the 390 apparent "fa4" hits are hex fragments in cache hashes (`row0=2d432779fa4d`). The main model runs its own QSA state backend, so the hd256 gate is never consulted. Cost: one grep. |
  | `v1/attention/backends/mla/prefill/flash_attn.py:392` | `device_capability[0] in (10, 11)` | MLA prefill — we do not run MLA. Low value. |
  | `models/inkling/nvidia/ops/fa4_rel_attention.py:33` | `capability.major in (10, 11)` | different model family, not ours. |
  | `model_executor/kernels/linear/__init__.py:1068` | `compute_capability in (100, 103)` | det-180. Real, but **inert for us** — no quantized dense Linear in Flash-Next, and the 27B is W4A4/compressed-tensors. |
  | `scaled_mm/pytorch.py:304`, two `self.arch == 90` in cute paged_kv | | sm_90 paths, not exclusions of us. |

  **Precedent that this is a normal change:** `models/kimi_k3/nvidia/kda.py:186` reads
  `capability.major in (9, 10, 12)` — someone already added a 12 arm in this codebase.

  **Method, and the part that is not automatable:** a gate excluding sm_12x is only a *bug* if the
  kernel would actually run there. That is per-site and needs a build or a test, which is why this is
  a scan plus judgement rather than a patch. Start with `fa_utils.py:233` because the head_dim
  matches; first question is whether FA4 is ever selected on our stack — `kernelroster.py` on any
  existing log answers that for free before anything is built.

  Re-run the scanner after every venv bump; gate lists move (det-180's file changed between 0.27.1
  and 0.28.1).

- **PLE mmap — UNGATED 2026-09-11 (det-193): cudagraphs capture NOTHING here (7 starts, 3 configs),
  so `--enforce-eager` costs nothing and the 3-file port is worth doing. NEXT UP.**
- ~~**Discriminator for det-193**~~ **DONE 2026-09-11 (det-194): speculation is NOT the cause — capture
  is still 0.0 GiB with `speculative_config=None`. Four candidates dead (eager, mode, sizes, spec);
  `splitting_ops` is the survivor and stays a hypothesis. Clean test if ever wanted: a plain dense
  model on the same venv — needs a second checkpoint, not a flag. Not blocking anything.**
- **PLE mmap as an alternative to CPU offload — could revive the ngram comparison (found 2026-09-07
  in [Radar105/qwen38-flash-next-nvfp4-spark](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark)).**
  det-160 established that `VLLM_PLE_CPU_OFFLOAD` forces a V1 conflict that makes ngram/ngram_gpu
  impossible on our stack, which is why the published page's ngram cells are unreproducible. Radar105
  runs the SAME model in production with `VLLM_QWEN4_PLE_MMAP=1` instead. **Verified: that env var is
  in neither our venv nor upstream main** — it is in their local patch ("the existing PLE mmap
  reader"), on a newer base (`7fbd44cb`, 2026-09-05) than our merge-base `d9105ea8`.
  **Steps:** (1) read `patches/vllm-complete.patch` and isolate the PLE mmap reader — do NOT apply the
  whole 23-file patch, it stacks five upstream PRs and local adaptations; (2) check whether the mmap
  path avoids the `uniproc_executor.py:71` / `parallel.py:491` V1 rejection; (3) if it does, it
  unblocks the ngram arms AND is a second PLE implementation to A/B against the offload worker (our
  offload takes 16 page faults per token — LOW-3). Requires a venv patch, so it waits for an idle box.
  **Also from that repo, no action needed but worth knowing:** they run `--enforce-eager` in
  production (independent support for det-158 — cudagraphs may be doing nothing here); they carry
  #53798 + #54076, the two still-open legs of our determinism chain, plus #54713 and #55390 which are
  new to us; and they do NOT carry #55122, so their production has the QSA top-k nondeterminism.
  Their decode numbers (26.8 tok/s @47.6k, 33.8 @30k, production median 22.7) agree with ours.

- ~~**NVFP4 kernel selection on sm_121 — REOPENED for the DENSE 27B**~~ **WITHDRAWN 2026-09-11: the
  27B is W4A4 with *dynamic* activations (no stored `input_scale`) on compressed-tensors, and 13 run
  logs show `FlashInferCutlassNvFp4LinearKernel`, never Marlin. The published Periodic Table is
  correct and needs no change. Closed for Flash-Next too:
- ~~**...for Flash-Next**~~ **CLOSED 2026-09-11 (det-180):
  real upstream, INERT for us — our checkpoint has zero quantized dense Linears (73,728 expert matrices
  + 1 PLE embedding, everything else excluded), so the selector is never consulted. The doubt over
  [[w4a16-vs-w4a4-measured]] is retired. Original text below.**
- **NVFP4 kernel selection on sm_121 (det-159, upstream #55397 /
  fix #55405).** VERIFIED in the prod venv: first match on sm_121 is
  `FlashInferCuteDslNvFp4W4A16LinearKernel`; three native W4A4 kernels sit below it unreached.
  Queued behind `mtprem`: apply #55405, confirm the selection flips, A/B prefill/TTFT + c=1 decode.
  **Then re-examine [[w4a16-vs-w4a4-measured]]** — if the W4A4 arm's dense linears ran through a
  16-bit-activation kernel, the 0.42 pp fidelity gap we measured is suspect (MoE path unaffected, so
  the finding is questionable, not dead). We have the hardware; the issue author appears not to.
  Triage of the other five new issues: **#55506** needs PP>=2 (we run PP=1) — does NOT affect us, but
  it is the same fault family as our #55375 (poisoned recurrent state, all-NaN logits, token 1023).
  **#55507** claims to fix #53142 — OUR bisection thread — and is the same territory as the still-open
  #53798; check whether our `cache_config.block_size` differs from `MambaSpec.block_size`.
  **#55514** touches only the grouped_topk PYTHON fallback; we use the fused kernel — does NOT affect us.
  **#55518** is a cosmetic warning about our own `disable_eagle_block_drop` flag — harmless, low value.

- **Triage the six new upstream issues in our areas** (found 2026-09-07, log 83). None actioned:
  **#55518** prefix-cache warning fires even with `disable_eagle_block_drop` (our flag — likely a
  cosmetic warning bug, cheap to confirm), **#55506/#55507** mamba spec-decode block tables and align
  state-index seeding (overlaps #53798, which is still open and one of the three legs of our
  determinism chain), **#55514** deterministic expert selection in `grouped_topk` (our determinism
  area — check whether it collides with our det finalize overlay), **#55397/#55405** NVFP4 kernel
  selection on SM12x (our hardware, W4A16 vs W4A4 ranking — relevant to
  [[w4a16-vs-w4a4-measured]]), **#55452/#55406** cudagraph capture/replay faults (adjacent to det-158).
  Also watch **#55314** for a reply.

- **Does prod capture cudagraphs at all? (det-158, queued, ~3 starts)** The capture-width A/B came
  back null because **neither arm captured graphs** — `0.0 GiB for CUDAGraph memory`, zero
  `Capturing CUDA graphs` tqdm lines while checkpoint-loading bars are present in the same logs.
  Add an `FN_CG_MODE=NONE` arm: if NONE == PIECEWISE == wide, cudagraphs are inert in this config and
  det-136's null has a second explanation. Until this settles, the MiaAI #19 commitment is NOT
  discharged and the capture-width sweep must not be re-run.

- **Permalink hygiene before every upstream post** (2026-09-07, log 82 correction). Four links in the
  #55122 routing comment 404'd because the SHA was captured before the commit that added the files.
  Capture the SHA *after* commit+push, then
  `curl -s -o /dev/null -w '%{http_code}' -L <url>` each one and require 200 before posting. A full
  audit of the #55122 body and all nine comments found only those four; everything else resolves.

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
- ~~**PR #55661 (swizzle gate)**~~ **CLOSED BY US 2026-09-07 09:57** (upstream log 63). The A/B answered
  it: the 7.5k control — where gate and merged pick the SAME swizzle, so it cannot be the variable —
  moved +0.94 %, the same as every changed cell. That is drift, not the gate. Posted "the objection was
  right — closing" with the six-arm table. The small-M island question is moot with the PR withdrawn;
  the hold-out result (island and slab-only both 236.3 pp on 88 held-out cells) stands as the record of
  why, and is worth reusing if the swizzle is ever revisited.
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
- **Secret scan must cover Modal tokens too.** The documented pre-commit scan
  (`git grep -nI -E "sk-[A-Za-z0-9_-]{16,}|develop8\.|BEGIN [A-Z ]*PRIVATE KEY" -- .`) does not match
  Modal's `ak-…` / `as-…` shapes. Use
  `git grep -nI -E "sk-[A-Za-z0-9_-]{16,}|\b(ak|as)-[A-Za-z0-9]{20,}|develop8\.|BEGIN [A-Z ]*PRIVATE KEY" -- .`
  Audited 2026-09-07 after the token was pasted into the session: **no literal token in any repo,
  working tree or history** (notes, vllm fork, flashdgx clone), none in `/tmp/claude-1000`,
  `/opt/llm/runners`, `/opt/llm/*.log` or `.bash_history`. The only copy is `~/.modal.toml`, which was
  mode **644** and is now **600** — servers here run as `uid=llm` and could read it. Rotate at
  modal.com/settings/tokens if the transcript ever leaves this machine.
- ~~The swizzle harness and its 223 MiB CUTLASS tree still live in an old session's `/tmp` scratchpad~~
  **DONE 2026-09-07**: harness rescued to `/opt/llm/runners/swzharness` (5.6 MB, world-readable, with a
  README). The CUTLASS tree was deliberately NOT copied — it was a pristine NVIDIA checkout at `cb42473`
  with zero local edits, so it is a one-line re-clone, not an artefact. Pin recorded in the README.
- `/tmp/claude-1000` is `drwx------ jschmied` and servers run as `uid=llm`: anything a server must read goes
  in `/opt/llm/runners`, world-readable. This silently invalidated four A/B arms on 2026-09-07.


## New upstream issues touching our stack — triage 2026-09-08 (37 scanned, created >= 09-05)

Nothing posted. Ranked by overlap with what we run and what we know.

1. ~~**vllm#55496**~~ **DROPPED 2026-09-08.** PRs #55498 and #55513 already cover it; #55513 adds
   `_remap_quantized_layers()` to `vllm/models/qwen4_exp/{nvidia,amd}/mtp.py` — the exact fix we had
   drafted — plus `FP8_PB_WO`/`FP8_BLOCK_SCALES` both. Our draft also wrongly claimed Qwen4Exp is
   out of tree; it is at `vllm/models/qwen4_exp/`. Draft deleted. Do not re-derive this.
   ~~**Original entry:** ModelOpt MIXED_PRECISION cannot load FP8_BLOCK_SCALES MTP experts~~
   (`nvidia/Qwen3.8-Flash-Next-NVFP4` + MTP). **Two DGX Sparks, GB10 `sm_121a`, and the same preview
   build `0.1.dev20073+g8e685d198` our Quant Map cites.** We hold direct knowledge here: memory
   `modelopt-quantized-layers-trap` — MIXED_PRECISION reads `quantized_layers`, NOT `config_groups`,
   and the wrong field yields a W4A4 kernel with no `input_scale`. Their failure is
   `mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`. **Highest-value place we could
   contribute, and the one where we are least likely to be wrong.**
2. **vllm#55600 — hybrid mamba prefix-cache hit reads out of bounds** (Xid 31): `add_request` seeds
   the state slot with `cache_config.block_size` *after* `_initialize_kv_caches` lowered it to the
   min over prefix-cacheable groups, which for a **DFlash2/EAGLE-style drafter group** is far below
   `mamba_block_size`. Adjacent to our own #55375 (strided `state_indices[:,0]` read with unit
   stride). We run DFlash2 + prefix caching + GDN. **Their trigger needs a sliding-window drafter
   group and our launcher sets no `sliding_window`, so the precondition may not hold for us — but
   the block-size-minimum mechanism is generic. VERIFY on the live config when the box is free**;
   the journal check attempted 2026-09-08 08:5x did not surface the block sizes.
3. **vllm#55766 — Qwen3.5/3.8 hybrid GDN: NaN logits after a prefix-cache hit** when the previous
   prefill ended 4–10 tokens past a block boundary, mamba cache mode `align`, v0.28.0. Our model
   family, our `align` mode, and the same area as #54076/#53798/#54173. Finding 76 already records a
   partial-vs-full prefix-cache-hit logits difference on this path — theirs escalates it to NaN.
4. **vllm#55524 — [RFC] Mamba2 exact-replay decode.** Carries a maintainer ruling that matters to
   our framing: **"batch invariance is organised per kernel, not per model"**, and the behaviour
   belongs under `VLLM_BATCH_INVARIANT=1` rather than a per-model opt-in. Read before writing
   anything further about batch invariance — we disclaim it in every determinism post.
5. **vllm#55775 — MTP + FlashInfer long-context CUDA IMA / Xid 31 on v0.27.1**, Qwen3.8-27B-NVFP4.
   Our model and our vLLM version (memory `vllm-027-cutover`).
6. **vllm#55581 — FlashInfer `get_cudagraph_support()` divides the target's head count by the
   drafter's.** Spec decode + cudagraph + FlashInfer; we are mid-cudagraph-A/B (`cgnone2`).
7. Lower: #55800 (DFlash2 sliding-window drafter admission deadlock — same precondition question as
   #55600), #55517 (`qwen3.8-flash-next` divisibility assert), #55515 (PLE embedding forces PP=1),
   #55580 (GDN 27B fp8 KV TP2 c32 −24 % step), #55569 (GLM-5.3-Flash 230K prefill exhausts unified
   memory on GB10).

## ~~Venv bump dev401 -> dev524~~ — DONE, and prod has been switched since (see CURRENT, 2026-09-20)

Target: main HEAD `5db652225`, wheel `vllm-0.28.1rc1.dev524+g5db652225-cp38-abi3-manylinux_2_28_aarch64.whl`
(311 MB, cached at `/opt/llm/runtime/wheels/`). Current serving venv: `8340fe1bb` / dev401, 04 Sep.
Recipe and its traps: `tools/main/BUILD-RECIPE.md`.

**Outcome: done.** fnmain3 built from the dev524 wheel into a clone, the two rejected hunks
hand-ported, and proved working (det-177: byte-identical output to fnmain2 on two of three shapes,
plus a real `PleOffloadWorker` process). Overlay regenerated as
`tools/main/fnmain-overlay-dev524.diff` — 12 files, re-applies with 0 failed hunks; it shrank from
17 because upstream now ships `vllm/v1/ple_offload/`. **Prod default unchanged.** The sizing that
led here:

**Dry-run result (`bumpdry`, prod untouched): 16 files check, 3 failed hunks, all in ONE file** —
`vllm/models/qwen4_exp/nvidia/ple_layer.py` (hunks 7 and 12). Everything else applies with 33
offsets and 2 fuzzy hunks. So this is a contained hand-port, not an overlay regeneration.

Upstream movement in the files we patch, vs our current venv:

| file | changed lines |
| --- | --- |
| `vllm/models/qwen4_exp/nvidia/ple_layer.py` | **684** |
| `vllm/v1/worker/gpu_worker.py` | 140 |
| `vllm/v1/worker/gpu/model_runner.py` | 127 |
| `vllm/model_executor/model_loader/weight_utils.py` | 69 |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | 33 |
| `vllm/models/qwen4_exp/nvidia/model.py` | 27 |
| rest | ≤ 24 each |

(`vllm/v1/ple_offload/*` and `ple_offload_layer.py` show as MISSING upstream because our overlay
*adds* them — expected.)

**Plan when given the go:** clone `fnmain2` → `fnmain3`, verify the interpreter rewrite by a server
log line and not by absence of error, install the wheel `--no-deps`, back up the pristine package,
apply the overlay, hand-port the 2 `ple_layer.py` hunks, re-run `prod_det_overlays.sh`
**against fnmain3** (it hardcodes fnmain2 — fix that first or it patches the wrong venv), then the
verification gate in the recipe. `fnmain2` stays intact; prod moves only by changing `FN_VENV`.

**Disk is the constraint:** 50 GB free at 95 % used; the clone is ~16 GB. `vllm-venv-fnmain` (16 GB)
looks like a stale previous generation but is the launcher's default when `FN_VENV` is unset — not
free to delete.

**Before bumping, finalise anything that must stay comparable.** det-169/171/172/173 were all
measured on `8340fe1bb`; vllm#55272 removes torch.compile for this model, so post-bump numbers are
a different execution model.

### ✅ The "files are in flight" banner is gone — verified 2026-09-20

The upload landed: 48 `layer*.safetensors`, 63.3 GiB, and the live card no longer contains the
banner string. Nothing to do.

## Levers queued 2026-09-22 (after findings 196-200)

Ranked by value / cost. None started; the blockers above them are being worked first.

1. **Slice the draft head in FP8 or NVFP4, not BF16.** `tools/draft_vocab/dv_patch.py` builds the
   32k slice as an *exact BF16 dequant* of the FP8_PB_WO head rows: 32k x 2560 x 2 = **164 MB read
   per draft token**. FP8 -> 82 MB, NVFP4 -> ~41 MB. We measured **+7.8%** (finding 198) from cutting
   248,320 -> 32,768 rows; this attacks the same projection again on bytes. Needs `torch._scaled_mm`
   for the FP8 path (`get_top_tokens` currently does a plain `F.linear` on a BF16 weight).
   **Cheapest remaining large win.**
2. **fp8 KV** — vllm#55557 merged 2026-09-16; our base is 2026-09-08. Measured ×1.72 KV pool on this
   box (`notes/fp8-kv.md`). Stacks with the drafter's +13.7% (finding 196) because they are
   different bytes. Gated on the upgrade blocker.
3. **Capture widths at concurrency.** `[4,8,12,16,20,24]` at `SEQS=6`. Finding 200 could not separate
   it single-stream, but above width 8 we capture nothing at all, so concurrent serving runs
   uncaptured decode shapes. Cheap: `FN_CG_SIZES` is a launcher env knob.
4. **MoE epilogue fusion** — findings 144/145, the 36.4% bucket sitting at the DRAM floor. Largest
   kernel-side prize, largest effort.
5. **65k vs 32k draft vocab** — one arm. Expectation is that 32k holds (99.6% coverage, +0.5 pp
   acceptance at 32k, det-135), so this is cheap disconfirmation rather than a likely win.
6. **`--dense` drafter A/B** — `mtpfp4d` is built and config-verified (shards 7.83 GB vs 7.92 GB).
   ~90 MB, ~+0.4% KV, no speed change predicted. Not worth 4 loads unless something else needs it.
