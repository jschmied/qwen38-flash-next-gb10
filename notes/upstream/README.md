# Upstream drafts — NOT POSTED

Drafts for one new vLLM issue and two comments, prepared 2026-09-02. Nothing here has been
posted. Post in this order, so the comments can link to a real issue number:

1. `issue-vllm-moe-fused-finalize.md` → **FILED 2026-09-02 as vllm#54945** (cost numbers labelled n=1).
2. `comment-vllm-54173.md` → superseded; a one-line cross-reference to #54945 was POSTED 2026-09-02.
3. `comment-flashinfer-3957.md` → POSTED 2026-09-02 (shortened to 3 sentences).
4. PR (optional, `pr-plan.md`)

Separate item: `pr-54552-body.md` — the QSA ring PR for #54552 (maintainer bojiang3 asked for it
   2026-09-02). Branch pushed to `jschmied/vllm:fix/qsa-ring-widen` (02001b44); opened as vllm#54912 on 2026-09-02.

4b. (kept): one kwarg + one env var; only after the issue has a number.

## Rules that apply (vLLM contributing docs, checked 2026-09-02)

- "Do not submit 'pure agent' PRs. The human submitter is responsible for reviewing all changed
  lines, validating behavior end-to-end, and running relevant tests."
- "Always mention when a pull request includes AI-generated code. Add a note in the PR description."
- Commit trailers: `Co-authored-by: Claude` plus DCO `Signed-off-by: <name> <email>` (`git commit -s`).
- The issue template requires `python collect_env.py` output (included) and a minimal repro
  (included). There is no separate AI rule for issues; the disclosure line is added anyway.

## Before posting (each item has cost us before)

- Re-read the target thread first; the comment drafts were written against the thread state of
  2026-09-02 (vllm#54173: 4 comments as of 13:20; flashinfer#3957: 0 comments).
- Search vllm issues for `use_fused_finalize` and `flashinfer cutlass moe deterministic` once more.
  Done 2026-09-02: no hit — the knob is absent on `main` (call site unchanged).
- The repo scripts carry the server key and the sudo password: nothing from `*.sh` goes into a
  post. Scan the draft: `grep -nE "sk-[A-Za-z0-9_-]{16,}|develop8\." notes/upstream/*`.
- Do not claim an MTP benefit. `MTPFIX_a` (fix installed, MTP n=5, 8-turn loop) landed in the
  slow regime — 67.25 ms/tok, acceptance 9.3 %, mean accept len 1.47 — so the deterministic
  finalize does not by itself stabilise MTP acceptance. b/c pending; the issue claims determinism
  and its cost only.

## What is deliberately left out

- The FlashInfer autotune-cache-key defect gets no issue of its own: `MoERunner.get_cache_key_extras`
  exists upstream from v0.6.18rc2 (absent at #3984 2026-08-06, present at #4106 2026-08-13). It is
  mentioned in the vLLM issue as a caveat for 0.6.17 users only.
- Local venv patches (FP8 KV for GDN, PLE-offload backport, hyperconnection/lm_head edits) are
  disclosed in one line; none touches the MoE call path, and the `emulation` control runs through
  the same patched build.

5. 2026-09-02 evening: acceptance-effect comment POSTED on vllm#53142, one-liners on #54076 and #53798 (finding 46).

6. **2026-09-03 — CORRECTION OWED.** The acceptance-effect numbers posted on #53142 / #54076 / #53798
   are contaminated by the `ignore_eos` artifact (finding 59). Draft: `comment-53142-correction.md`;
   post after `MTPGRID3` (+ one unpatched EOS-correct run for the "before" number). The +3.6 % / +17 %
   decode costs in #54945 / #54948 / flashinfer#3957 are no-spec token-rate comparisons at equal
   token counts and are not affected by the filler; they remain n=1 as labelled.
7. Kernel fix for #54521 (deterministic `persistent_topk`, all five emission sites):
   `patches/kernel-det/` — compiles for sm_121a; standalone `_C_det` link + reference tests queued.
   PR after the tests pass and a prefill A/B against the exact-selection workaround. Title idea:
   `[Kernel] Make persistent_topk deterministic (index-ranked ties, sorted rows)`; link #54521,
   #51782, our finding 53/54 chain as evidence; disclose AI assistance; `git commit -s`.

8. 2026-09-03 10:xx: correction POSTED on #53142 (https://github.com/vllm-project/vllm/issues/53142#issuecomment-5522457527) and one-liners on #54076/#53798; the unpatched EOS-loop number is still owed (mtpgrid0).
9. 2026-09-03 12:xx: kernel PR **OPENED** — https://github.com/vllm-project/vllm/pull/55122 (`[Kernel] Make persistent_topk deterministic`, Fixes #54521; body in `pr-topk-det-body.md`). Sources = `patches/kernel-det` v2.3 + `RADIX_THRESHOLD` 16384, clang-formatted; new pytest cases in `tests/kernels/test_top_k_per_row.py` (copy: `patches/kernel-det/test_upstream_top_k_per_row.py`, run against `_C_det` via `detplugin.py`). OWED on the PR: the pytest result (`kpytest`, queued) and the end-to-end decode/TTFT A/B (`kdetab`, queued).
   - 12:xx: related-issue sweep. #51782 (xueyangcs: persistent_topk silently DROPS candidates on coarse-bin overflow, 4096/3708 buffers; Leonccaa confirms Flash-Next rows=1 nondeterminism there) — our PR removes the buffers → added `Fixes #51782` + a narrow-value-range test. #53287 (LopezCastroRoberto, OPEN, exploratory, no reviews, merge conflicts: widened histogram + exact overflow fallback, keeps buffers) — named in the body as the alternative mechanism. #54739 (Thai corruption) — reporter shows it is NOT block selection; not linked. #54513 MERGED 09-02: the Flash-Next model dir moved to `vllm/models/qwen4_exp/` on main (our local patches target the preview tree). No comment posted on #51782 — needs the user's go.
10. 2026-09-03 12:5x (user's go "you can ping crimsonjoo's"): pointer to PR #55122 POSTED on the original recipe issue https://github.com/blazux/qwen3.8-Flash-DGX/issues/3#issuecomment-5524669381 (k3dani's report, closed 08-28 with the exact-top-k default; crimsonjoo's README is a copy of blazux's) and as a new issue on the copy https://github.com/crimsonjoo/DGXspark1-Qwen3.8-Flash-Next/issues/1. Text: `comment-blazux-3.md`.
11. 2026-09-03 17:3x (user: "post it as short comments"): reply to k3dani's GB10 validation POSTED on PR #55122 (https://github.com/vllm-project/vllm/pull/55122#issuecomment-5528191146, text `comment-55122-reply-k3dani.md`: same set, different output order, A/B follows) and the PLE-gate pointer POSTED on #54765 (https://github.com/vllm-project/vllm/issues/54765#issuecomment-5528191382, text `comment-54765-ple-gate.md`; offered a PR). Still owed: the e2e A/B on #55122 (runners lost in the 17:08 reboot, parked) and the #53142 correction numbers (start c of the grid).
12. 2026-09-03 18:5x (user: "start with proposed 1 and 2"): M-chunk PR **OPENED** — https://github.com/vllm-project/vllm/pull/55174 (`[Kernel] Chunk M for the CUTLASS blockwise FP8 GEMM on SM 12.x`, body `pr-fp8chunk-body.md`, findings 67/71/73; 8/8 bit-identical on GB10; branch `fix/sm12x-blockwise-fp8-m-chunk` on the fork, worktree `~/git/vllm-fp8chunk`).
   - 18:4x (user: "can we remove PR and create a new one later?"): #55174 CLOSED with a one-line note (Python-level loop specialises Dynamo; family gate too broad — the user's review). The C++ version (loop inside `cutlass_scaled_mm_blockwise_sm120_fp8`, L2-based gate) is on branch `fix/sm12x-blockwise-fp8-m-chunk` in `~/git/vllm-fp8chunk` (uncommitted), standalone validation pending the box; new PR after that.
13. 2026-09-03 18:5x (user: "open a new one as draft"): **DRAFT PR #55180** — https://github.com/vllm-project/vllm/pull/55180 (`[Kernel] SM 12.x: chunk M in the CUTLASS blockwise FP8 GEMM when the weight exceeds the L2`; branch `fix/sm12x-blockwise-fp8-l2-chunk`, one commit `587c6474`; C++ loop inside the op, L2-based gate, compiled dynamic-M test). Owed on it: the standalone GB10 validation (bit-identity + throughput of the C++ path, `scratchpad/chunk/`), then un-draft.
14. 2026-09-03 19:3x: PR #55180 **READY FOR REVIEW** (un-drafted after the standalone GB10 validation, finding 75; commit `027e31e0`: K-aware chunk rows, 1.5-chunk threshold, tests mirror the rule, 48/48 identical). Still owed: a server-level TTFT number through the C++ path at batch ≥ 8192 (nice-to-have; the batch-4096 prod config does not chunk).
15. 2026-09-03 19:3x (user: "post"): GB10 validation of the offload worker POSTED on vllm#53899 (https://github.com/vllm-project/vllm/pull/53899#issuecomment-5529563866, text `comment-53899-gb10-validation.md`) and the reply to k3dani's write-up POSTED on PR #55122 (https://github.com/vllm-project/vllm/pull/55122#issuecomment-5529564166, text `comment-55122-reply-k3dani-2.md`).
16. 2026-09-03 19:5x: CodeRabbit round on #55180 addressed in `48af7c54` — K==0/M==0 guard before the chunk-row division (was a divide-by-zero on empty K), the 147 MB test case gated to SM 12.x (its fp32 baseline is ~1.2e12 ops), docstring for the input helper. Standalone rebuilt with the guard: 6/6 identical.
17. 2026-09-03 20:3x: CodeRabbit round on #55122 addressed (commit on the branch): launcher reads device props per call (the static cache was shared across devices; my smem cap depends on it), k=1024 added to the exactness matrix. Standalone `_C_det` rebuild + re-test queued for a GPU-free slot (`kdetrebuild`, after the night chain).

18. 2026-09-04 09:03 — #53142 follow-up: complete three-start acceptance ladder (`comment-53142-followup-grid.md`) → https://github.com/vllm-project/vllm/issues/53142#issuecomment-5536961342
19. 2026-09-04 09:03 — PR #55122 follow-up: server-level A/B, no TTFT/per-turn cost, det==exact 6/6 (`comment-55122-followup-e2e.md`) → https://github.com/vllm-project/vllm/pull/55122#issuecomment-5536961535
20. 2026-09-04 11:17 — #54928: same-symptom data point (block path == q_len=1 path bit-for-bit after the three kernel fixes), two diagnostics, GB10 trace offer (`comment-54928-offer.md`) → https://github.com/vllm-project/vllm/issues/54928#issuecomment-5538340562
21. 2026-09-04 13:33 — blazux issue #3: M%4 pad drop-in + numbers, #53388 port with the −26 % table, #55180 correction (`comment-blazux-4.md`) → https://github.com/blazux/qwen3.8-Flash-DGX/issues/3#issuecomment-5539850457
22. 2026-09-04 13:33 — #53670: hybrid + in-checkpoint-MTP evidence table for the trailing-block drop (`comment-53670-hybrid-evidence.md`) → https://github.com/vllm-project/vllm/issues/53670#issuecomment-5539850703
23. 2026-09-04 15:24 — PR #55180 reply to gau-nernst: swizzle vs chunking not exclusive; will bench their Triton/CuteDSL FP8 kernels on GB10 (`comment-55180-reply-gaunernst.md`) → https://github.com/vllm-project/vllm/pull/55180#issuecomment-5541081489
24. 2026-09-04 16:11 — PR #55180 REWRITTEN (ff29cfc4 pushed to jschmied/vllm, rebased on the fork's merge commit): chunking → tile-scheduler max_swizzle_size=8 behind the same L2 gate; title+body v2 (`pr-55180-body-v2.md`); reply to gau-nernst with the sweep (`comment-55180-swizzle.md`) → https://github.com/vllm-project/vllm/pull/55180#issuecomment-5541663159

Gotcha (2026-09-04): `gh pr edit` on vllm-project/vllm fails with a "Projects (classic) is being deprecated" GraphQL error and
silently changes nothing — edit title/body with `gh api -X PATCH repos/vllm-project/vllm/pulls/<n> --input body.json`. The PR
branch's push remote is the fork (`jschmied/vllm`), and GitHub's "update branch" adds merge commits there that a local
branch lacks — fetch + rebase before pushing.
25. 2026-09-04 16:22 — PR #55180 follow-up commit (activation-bytes gate, balanced reference slices, comment); body v3; comment → https://github.com/vllm-project/vllm/pull/55180#issuecomment-5541812440
26. 2026-09-04 21:5x — MiaAI-Lab single-Spark kit, issue #4: MAX_NUM_SEQS ceiling, prefill config on the same image (~1.8× TTFT), warm-turn fixes (`issue-miaai-single-spark.md`) → https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/4
27. 2026-09-04 22:2x — #54521 reply to davidcanar (gfx1151/GLM DSA): sequential-vs-concurrent discriminator, three candidates, GEMM M-invariance table (`comment-54521-davidcanar.md`) → https://github.com/vllm-project/vllm/issues/54521#issuecomment-5545684729
28. 2026-09-04 22:4x — **RFC OPENED** #55394: tile-union QSA prefill kernel (design, GB10 numbers, four maintainer questions, feedback until 09-11; `rfc-qsa-tile-union.md`) → https://github.com/vllm-project/vllm/issues/55394
29. 2026-09-04 23:4x — #54521 reply 2 to davidcanar: recurring hashes = reduction order, the indexer fault deserves its own issue, accept the ROCm script PR (`comment-54521-davidcanar-2.md`) → https://github.com/vllm-project/vllm/issues/54521#issuecomment-5546669266
30. 2026-09-04 23:4x — MiaAI #4 reply to malvavisc0: quoting bug makes the compile row unmeasured, prefix-cache caveat (`comment-mia4-malvavisc0.md`) → https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/4#issuecomment-5546669405
31. 2026-09-05 07:5x — **PR OPENED** #55430 (tile-union QSA prefill kernel, SM121; 2 commits on 8369affa; body `pr-qsa-tile-union-body.md`) → https://github.com/vllm-project/vllm/pull/55430
32. 2026-09-05 07:5x — RFC #55394 reply linking the PR → https://github.com/vllm-project/vllm/issues/55394#issuecomment-5549779240
33. 2026-09-05 10:5x — #54521 CORRECTION: our blockwise-FP8 M-invariance row withdrawn (scale-layout artifact, finding 122); GB10 rerun pending (`comment-54521-correction.md`) → https://github.com/vllm-project/vllm/issues/54521#issuecomment-5550651254
34. 2026-09-05 11:2x — PR #55180: pushed f945fa34 (tests folded) + af52cd77 (gate = weight > L2 only) and replied with the 10-shape sweep (`comment-55180-reply-gaunernst-2.md`) → https://github.com/vllm-project/vllm/pull/55180#issuecomment-5550814597
