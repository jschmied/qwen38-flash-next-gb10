# Upstream drafts — NOT POSTED

Drafts for one new vLLM issue and two comments, prepared 2026-09-02. Nothing here has been
posted. Post in this order, so the comments can link to a real issue number:

1. `issue-vllm-moe-fused-finalize.md` → new issue on vllm-project/vllm (bug template).
2. `comment-vllm-54173.md` → comment on vllm#54173 (same build, same box, different symptom).
3. `comment-flashinfer-3957.md` → comment on flashinfer#3957 (different kernel, same suspect).
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
  2026-09-02 (vllm#54173: 0 comments; flashinfer#3957: 0 comments).
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
