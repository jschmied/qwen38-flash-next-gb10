# Open work, ranked

Cleaned 2026-09-24. Everything closed, superseded or historical moved verbatim to
[`TODO-archive-2026-09-24.md`](TODO-archive-2026-09-24.md). Findings live in `prefill-investigation.md` /
`determinism-investigation.md`; this file lists only what is still open.

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
6. **MoE epilogue fusion** (findings 144/145). This is the 36.4 % bucket at the DRAM floor: the biggest kernel prize
   and the biggest effort.
7. **65k vs 32k draft vocab.** One arm; cheap disconfirmation, and 32k is expected to hold.
8. **Finish the MTP re-measurement** (`mtp-remeasure-plan.md`). The Quant Map page flags its depth curve as under
   re-measurement.

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
decode-written blocks, 0 divergent. Keep watching; a comment with the MTP counter-datapoint could help, but needs a
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
