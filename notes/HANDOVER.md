# Handover — 2026-09-08, end of a long session

Written because the session ran long and produced an unusual number of self-corrections. Everything
below is committed; nothing important lives only in a chat log.

## The one thing to know

**PR #55122's kernel, on its own, does not deliver end-to-end reproducibility on this stack.**
`isolate4` (running at handover time) turns each of the four determinism fixes on alone against a
no-fix control:

| arm | disagreeing positions | note |
| --- | --- | --- |
| `none` | 325 | control misbehaved, so the run is valid |
| `qsadet` (**our PR #55122**) | 348 | no improvement |
| `cachekey` | 311 | no improvement |
| `plefix` | 282 | no improvement |
| `detfin` | died | not runnable alone — needs `cachekey` (the fixes are coupled) |

**No single fix comes near zero; all four together give exactly 0.** So the four are *jointly
necessary*, and our kernel is one component of a set rather than "the" fix. The per-arm numbers are
provisional — that run shared one `FN_CACHE_ROOT` across arms, against our own guidance; `isolate5`
redoes it with per-arm roots, `detfin+cachekey` paired, and an all-four positive control in the same
run. Quote det-181 (all four vs none, a clean two-arm comparison) until then.

For contrast, det-181: all four together give **exactly 0**, while true stock gives 335 disagreeing
positions with 10.63 spread from position 2.

**Consequence, and the first thing to act on:** the live #55122 body says *"Fixes #54521"* and frames
the kernel as what makes greedy decoding reproducible. On this evidence that is too strong. If
`plefix` turns out to be the load-bearing fix, say so on the thread **before** a maintainer merges
#55122 on the strength of a reproducibility claim. The kernel change is still correct and still
worth merging — as a kernel correctness fix with a measured performance profile, not as the fix for
#54521.

## Where things stand

**Upstream, ours:**
- **#55122** — head `7cfd04a39`, two perf commits added today (blocked 4-item emission;
  `RADIX_THRESHOLD` → 22,016). Ratio vs stock 0.72–1.78× over 43 cells, at or below stock on 27.
  Reviewed by LopezCastroRoberto 09-08; we replied conceding the accuracy point, dropping the
  change-the-default position, and offering our kernel behind their opt-in flag.
- **#55872** (theirs) — tested at their request: **does not start on sm_121**
  (`TopKRaggedTransform … operation not supported` at engine init). Their patch applies cleanly.
- **#54521** — ZC502's client-mode collector validated here (12/12 runs). Reply drafted, **not sent**.
- **#53899 + PR #13 on its fork branch** — the PLE offload subsystem and our semaphore fix. **Neither
  is upstream**: zero `vllm/v1/ple_offload/` files in the dev524 wheel, 404 on vllm main, #53899 open
  and `mergeable_state: dirty`.

**Three conflicting "the fixes" lists are in circulation** — see `notes/TODO.md` → Live state.
Reconcile before the next post.

**Box:**
- prod default venv is still `vllm-venv-fnmain` (`serve-fnmain.sh`). **Cutover to `fnmain3` is a
  one-line `FN_VENV` change and needs the user's go.**
- `fnmain3` = dev524 + our overlay, built and proved today (det-177): byte-identical output to
  fnmain2 on two of three shapes, real `PleOffloadWorker` process.
- `fnmain2` = dev401, what every measurement today used. Overlays currently ON
  (`qsadet=1 detfin=1 cachekey=1 plefix=1`).
- **Disk 34 GB / 97 %** — the tight resource.

## Traps that cost time today (all now in method.md / BUILD-RECIPE.md)

1. **Verify the control arm actually misbehaved**, not just that the knob moved. Five runs measured
   nothing (`cgsize2`, `cgnone2` c≥4, `thr`, `vpp4`, `vpp5`).
2. **`FN_DET_TOPK=0` is not "stock."** Only `qsadet` is env-gated; `cachekey` and `plefix` are not
   gated at all and `detfin` defaults on. A true stock arm needs `prod_det_overlays.sh off`.
3. **A `--no-deps` pip install is not a reset.** It overwrites the intersection with the wheel, so
   files we *add* survive. Diff against the extracted **wheel**, never a tarball of a populated venv.
4. **`patch --dry-run` prints "checking file"**, not "patching file" — a reject-count script reports
   a clean apply when it matched nothing.
5. **Check units with the tokenizer.** `chars/4` put a "2,600-token" prompt at 1,874.
6. **The response field is `message.reasoning`**, not `reasoning_content`, and `reasoning_tokens`
   can exceed a small `max_tokens`, leaving `content` empty.
7. **systemd `Environment=` with a space** needs quoting for systemd, or the unit silently never
   starts — and `systemd-run` failing is not the same as the server dying.

The pattern behind most of today's errors: **trusting something derived instead of the primary
source.** The wheel, the GitHub API and the tokenizer each answer in seconds.

## Scheduling — LOST when the session ends, recreate

Session-only crons: an hourly watchdog at :11 plus one-shots. Recreate the hourly one from
`notes/TODO.md` → Live state, and the standing rules: if the box is idle, take the next task without
asking; after starting any run, schedule a one-shot just after its expected finish stating what the
run decides and what each outcome means.

## Immediately next

1. Harvest `isolate4` (`/opt/llm/runners/results/isolate4.txt`) → finding after **det-182**. Read
   `none` first: non-zero or the run is void.
2. Re-run `tiecensus` (fixed: its output file must be `chown llm:llm`, the server runs as `llm`). It
   measures how often real indexer scores tie at the top-k boundary — the mechanism behind all of
   this, and still unmeasured.
3. Then: the #55122 correction, the ZC502 reply, MiaAI #19 (needs a wide-capture arm), `zsign`'s
   theirs-arm (#55314 needs its own driver).
