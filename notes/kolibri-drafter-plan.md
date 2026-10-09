# Kolibri-1 drafter: results so far and plan

Kolibri-1 (Aleph Alpha, 78B MoE, 3.5B active, FP8) ships no MTP head and no draft model exists for it. We serve it on
our TensorFold family `kolibri1` (PR #328) with copy drafting from the context. This note tracks the learned drafter.
Last updated 2026-10-04 15:40 (tonight section 15:40).

## Where things stand

Held out: the last 3 of Kolibri's own SWE trajectories (40,748 assistant tokens). Two measures:
- **held-out check** during training: along the conversation's text, step-1..3 top-1 against Kolibri's own choice, and
  mean drafts accepted in a row at depth 3 (`accepted`);
- **greedy acceptance** after training (`accept.py`): Kolibri decodes greedily from each assistant turn; mean drafts
  accepted per round at depth 1-6 (3,249 rounds). The comparable number.

| Run | Drafter | Data | Held-out step 1 | accepted (depth 3) | Greedy acceptance d1 / d2 / d3 |
|---|---|---|---:|---:|---|
| 1 | 1 layer, 71M, last state only | 7M SWE tokens (EAGLE-1 loss) | — | — | — |
| 2 | same, continued | + chat, 3.6M tokens | 65.4 % | — | 0.58 / 0.805 / 0.889 |
| 3 | same, 3-step rollouts (EAGLE-3 training-time test) | SWE + chat | 64.8 % | 1.267 | 0.576 / 0.836 / 0.950 |
| 4 | 2 layers, grown from run 3 | SWE + chat | 64.6 % | 1.241 | stopped at step 1,250: no gain |
| 5 | 4 layers, MLP 8192, 389M, taps 12/25/49, from scratch | SWE + chat | 54.5 % (step 2,500) | 0.956 | stopped at step 2,500: behind run 3 at half its data |

Findings:
- First-guess agreement stops at about 65 % whatever the size (runs 2-4): depth is not the limit.
- Rollout training lifts later steps (conditional step 2: 39 -> 45 %, step 3: 37 -> 44 %), well short of the 60-70 %
  in the EAGLE-3 / HASS papers.
- Greedy path (58 %) is below the text path (65 %): the drafter learned mostly other models' text.
- Estimated value of run 3 in serving, depth 2 with a 32k draft-head slice: about +25 % single-stream decode (verify
  cost per row not yet measured). Copy drafting gives +13 % (chat) to +31 % (greedy code).

## Field: 0xBakeer's TandemLLM Kolibri branch (stopped 2026-10-04)

`github.com/0xBakeer/TandemLLM`, branch `kolibri-experimental`, `docs/kolibri.md` (archived, not maintained):
- Block drafter (DSpark-style, 5 layers, 452M, taps from 5 layers) from scratch on 7,136 of Kolibri's own answers
  (6.6M tokens, one pass): 1.47 tokens a round at chain 3 on held-out answers (gate 2.0). If 1.47 includes the target's
  token, that is 0.47 accepted drafts against our 0.95 (greedy depth 3) — different held-out sets, not a measured
  comparison.
- Their measured decode-vs-prefill tap difference: 5 % relative RMS at layer 49 (our recordings use decode states).
  Tap scales differ widely (row RMS 0.53 after layer 1, 28 after layer 49, one channel 805).
- Verify cost per row on one GB10 (their NVFP4 engine, 1k context): 1 row 12.5 ms, 2: 16.5, 4: 23.2, 8: 35.0,
  16: 57.9, 32: 101.9 — each row brings its own experts; short chains (2-3) are the optimum.
- Lookup drafting with StairCut: 3.3x on edits/copies, 1.04-1.11x on code/chat with tools, 0.97-0.99x on prose/German.
- Decode 80.4 tok/s at 1k (NVFP4 requant, 2.44 GB/token, 72-74 % of the byte ceiling; lane-accumulator kernels, PDL,
  L2 prefetch) against our 40.5 (FP8 as released, 3.8 GB/token, ~56 %): the kernel-efficiency part is an engine lever.

Engine review (code read 2026-10-04, commit 4d17bd82; ranked by expected gain, all engine-side, all row-invariant):
1. PDL on every decode kernel, and each kernel prefetches its own weight rows into L2 (`prefetch.global.L2`, up to 8 KB
   a row) before `gdc_wait`. MoE order router -> shared expert -> routed experts, so the routed kernel already knows its
   expert id and prefetches that expert before its wait (`kernels.py:88-105, 711-731`). Timing only, no bit change.
2. One-row matmuls: 4-32 output rows a program, BK=512, ~200-1,900 programs, no `tl.dot`, no split-K (tried, dropped).
   Their FP8 GEMV gains nothing from lane accumulators (`kernels.py:563`): for our FP8 the lever is the shape, not lanes.
3. ~13 kernels a layer, one CUDA graph per token: q/k norm + RoPE + cache write in one kernel, split decode attention +
   combine, `r + rms(y)w1 -> rms(r')w2` fused with the 6 routed partials + shared expert, SwiGLU inside the shared down
   projection, BF16 router GEMV, one top-6 kernel. 39.4 -> 70.8 tok/s for this step (mixed with an attention requant).
4. Verify twins bit-equal to decode: one program per row in decode's order, rows adjacent so tiles hit L2; MoE pairs
   counting-sorted by expert; draft k/v in a side buffer until commit. Load-time `selfcheck()` (12 tokens as chain and
   tree, spec off on any differing bit) — copy it.
5. Expert-union ramp for R verify rows: 1, 1.2, 1.53, 2.06, 2.88, 4.04x one token's bytes at R = 1..32; their measured
   verify cost is worse than that model (7.7x at 32 rows against 4.04x).
6. Drafter: `--tap-norm` (fold per-channel tap RMS into `fc`) and a decode-vs-trainer tap check (`kd_tapcheck.py`).
   For us: run 6 trains on decode-path recordings (kind 1) but the held-out set is prompt-path rows; check that our
   tap bits match between the two paths before reading run 6's held-out number as serving acceptance.
Not for us: their NVFP4 weights, e2m1 tricks, prefill (2,170 tok/s, slower than ours), their NVFP4-tuned verify prices.
Details and file:line list: subagent report, kept in the session transcript; clone at the session scratchpad.

## Layer probe (2026-10-04 15:30)

Ridge probe from each layer's output at t to Kolibri's final state choosing token t + k, top-1 through Kolibri's head,
on held-out assistant tokens (`tools/kolibri/drafter/probe.py`, `notes/data/kolibri/probe/probe.log`):

| Layer(s) | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| 9 | 41.5 | 28.4 | 24.0 |
| 24 | 38.0 | 26.3 | 22.2 |
| 44 | 83.0 | 35.2 | 25.8 |
| 47 (best single) | 93.6 | 36.5 | 26.9 |
| 49 (final) | 99.5 | 35.1 | 25.7 |
| 34+44+49 (best set, k=2) | 99.6 | 37.0 | 26.7 |
| 9+24+49 (best set, k=3) | 99.6 | 35.8 | 27.3 |
| 12+25+49 (run 5) | 99.6 | 35.6 | 26.3 |

Information about later tokens sits in the late layers (44-48), not the middle; three layers add only 1.5-2 points
over the final state. Inputs are not the main limit either: data is. Taps for a warm-started try: 44/47/49 or 34/44/49.

## Tonight (2026-10-04): own data, then training from serving recordings

- Server `fx-kolibri-serve`: 6 streams, 131k context. Generating Kolibri's own data: the SWE Verified-30 run resumed on
  x86 (`/root/fullrun-kolibri-py-resume.sh`, 22 instances left) and chat answers to first user turns from the four
  downloaded sets (`fx-kolibri-genchat`, `tools/kolibri/drafter/gen_chat.py` -> `~/kolibri-drafter/own_chat.jsonl`).
- Recording (TF branch `kolibri1-record`, local, f94e0f2): `TENSORFOLD_KOLIBRI_RECORD=<dir>` makes the server write each
  kept row's token, layers 44/47/49 and Kolibri's top-32 log-probs (~15 KB a token, pauses below 25 GB free disk).
  Replies are unchanged (tested); prompt rows equal the prefill's states bit for bit; rejected drafts never recorded.
- TensorFold code for this (taps, recorder, row kinds): branch `jschmied/TensorFold:kolibri1-record` (88abeb0), on top of
  PR #328's branch; not part of the PR.
- Rows carry a kind (1: Kolibri generated it; held-out: the assistant mask). Training and the held-out check use only
  chains whose drafted tokens are kind 1 — the same assistant-only counting as run 3's 1.267 (an external review caught
  that the first version counted every row).
- `fx-kolibri-switch` (root) waits for the SWE generation to finish, then (aborting to a normal, non-recording server if
  the held-out recording fails `check_heldout.py` or the recording server does not come up): records the held-out set offline
  (`record_heldout.py` -> `~/kolibri-drafter/rec/heldout`), restarts the server with recording on (`rec/live`) and
  starts `train_rec.py` beside it (`fx-kolibri-drafter-rec` -> `out6/`, `train6.log`): no Kolibri copy, soft CE to
  the top-32 + head-state L1, 3-step rollouts, warm start from run 3 (fuse passes layer 49 through: step 0 reproduces
  run 3 exactly), held-out check every 250 steps. Logs: `~/kolibri-drafter/switch.log`.

Decision gate for run 6 (training on Kolibri's own recorded traffic): greedy acceptance at depth 3 on the same 3,249
rounds (`accept_mt.py`, needs a short server stop). From 0.950 (run 3) to >= 1.15: go to serving costs and engine
integration; ~1.25 closes the gap to the text path; <= 1.0: stop tuning the network, keep it as a fallback behind copy
drafting and look at trees or block drafting.

## Run 6 result (2026-10-04 19:20): memorised, stopped

Switch fired 18:01, all checks passed (held-out 36,000 rows, 5,397 assistant chains); training ran 18:03-19:17.
- Held-out (SWE, prompt path) fell every eval from step 250: 1.265 (step 0 = run 3) -> 1.146 -> 1.000 (2000)
  -> 0.817 (6250). Training soft CE 3.26 -> ~0.7.
- Cause, measured: data starvation. Recording gave 88k rows in 1.2 h (54 chat conversations, 2 generation workers);
  training read 7.3M rows in that time, ~83 passes. On those seen conversations run 6 scores 2.28 accepted
  (step-1 86 %) against run 3's 0.60: memorised.
- Run 3 on chat decode-path rows: 0.60 accepted, against 1.265 on SWE prompt-path rows. Domain or tap path (TandemLLM
  saw 5 % decode-vs-prefill tap difference) — open; fresh-conversation test (run 3 vs run 6 on conversations recorded
  after the stop, `eval_rec.py --after`) separates memorising from domain.
- Fresh-conversation test (12 chat conversations recorded after the stop, 16,021 chains): run 3 0.645, run 6 0.611.
  Run 6 learned nothing that generalises: pure memorising. The chat-vs-SWE gap (0.64 vs 1.265) stays open: domain or
  tap path. A SWE run served with recording on settles it: run 3 on fresh SWE decode-path rows near 1.2 = tap path
  fine (chat is just harder to draft); near 0.6 = decode taps differ from prefill taps.
- SWE run (x86, 30 instances, Kolibri on TF): 25 completed, 20 resolved.
- Fix before any rerun: cap passes (train only while rows seen < ~2x unique generated rows, else wait), an
  in-distribution held-out (every 10th recorded conversation), and enough recordings first (genchat runs until ~06:00).

## Run 7 (started 2026-10-04 19:55): pass cap 2, live held-out, SWE + chat recordings

- Tap path settled: run 3 on SWE decode-path recordings (404 requests, 132,013 chains, the new SWE run) accepts
  1.188 (step-1 0.681) against 1.265 (step-1 0.637) on the SWE prefill-path held-out. Decode taps are fine; chat's
  0.64 is the domain (chat at T=1 is harder to draft).
- More SWE data: x86 runs every instance whose image is cached and Kolibri has not run (`/root/fullrun-kolibri-more.sh`,
  unit `kolibri-swe-more`): Verified python-10, then Multilingual-105; temp 0.6 overlay (same as every SWE score).
- SWE score so far (Verified-30, same overlay and harness as the Qwen runs): Kolibri 20/30; Flash-Next 24-28 (mostly
  27); Qwen3.6-35B NVIDIA 19 (23 after infra reruns), Unsloth 17. Kolibri's empties: 3 hit our 131,072 server context
  (113-162 steps), 1 step limit, 1 repeated format error.
- Recording rate with SWE + chat: 5.5 GB per 20 min (16.5 GB/h), 125k generated rows per 20 min (1/3 of rows; prompt rows
  are the rest). 77 GB free at 19:44: the recorder's 25 GB floor is reached in ~3 h.
- Run 7: from run 3, `--passes 2 --live-held 10`, LimitNOFILE raised (each request is a recording of 5 memmaps; the
  default 1,024 descriptors crashed the first start). Step 0: SWE held-out 1.265, live held-out (58 requests, chat + SWE)
  0.966. Gate: live held-out above 0.966 without the SWE held-out falling below ~1.2.
- Run 7 trajectory: SWE held-out 1.265 -> 1.313 (250) -> 1.331 (500) -> 1.283 (750) -> 1.369 (1000): rising, unlike
  run 6. Live held-out 0.966 -> 1.459 (250) -> 1.519 (1000): too fast to be learning — the split is per request, and an
  SWE conversation's turns repeat each other, so held-out turns resemble trained turns. Upper bound only; the SWE
  held-out is the gate. The pass cap binds (training waits for recordings).
- Disk (2026-10-04 evening): freed qwen38-27b-fp8 (archived, re-verified), superseded venvs v030 + plepr (patches kept in
  /opt/llm/runners/venv-relics/), archived + freed exl3-3.05bpw and mtpfp4-plebf16 (`/opt/llm/runners/archmore.sh`).
- Recordings are backed up, not deleted (user: "better backup training data, if we need another run"):
  `recsync.py` (unit `fx-kolibri-recsync`, log `~/kolibri-drafter/recsync.log`) mirrors ~/kolibri-drafter to
  PBS `/mnt/bulk/gb10/kolibri-drafter` every 15 min; below 60 GB free it removes the oldest recordings whose every file
  matches PBS by sha256 (never the live held-out or rec/heldout).
- German (user 2026-10-04 21:00: "yes" to a German held-out + more German): Kolibri answers German prompts in German
  (103/123) and mostly reasons in German (94/123); German was 12.6 % of recorded generated rows (57 of 1,042 requests).
  Training resumed from run 7 step 2000 (`--init out7/drafter-step2000.pt --seen-gen 1055317`: the pass cap keeps
  counting; the fuse is kept when taps match) as `fx-kolibri-drafter-rec7b` -> `train7b.log`, with `--de-held 4
  --de-after <restart>`: every 4th German request recorded after the restart is held out (`held_de`), so the number is
  clean of run 7's earlier training. Chat generation split into one mixed worker (`own_chat.jsonl`) and one German-only
  worker (`fx-kolibri-genchat-de`, `own_chat_de.jsonl`, the two German sets), still two streams beside SWE's four.

## CORRECTION 2026-10-04 21:50: run 7 does not generalise; its SWE held-out gain is django

Supersedes "rising, unlike run 6" above as a verdict. Fixed checkpoints scored on recordings finished AFTER the
checkpoint was saved (clean by construction), `eval_rec.py --after`:

| fresh data (after 20:58, step-2000 save) | run 3 | run 7 step 2000 |
|---|---|---|
| SWE, non-django repos (138,692 chains) | 0.980 | 0.974 |
| chat (50,139 chains) | 0.665 | 0.677 (step-1 0.456 vs 0.453: noise) |
| SWE held-out (prefill, 3 django instances) | 1.265 | 1.382 |

- The SWE held-out is three django instances; the training recordings include the Python-10 slice's four django
  instances. Its +9 % is repo/content overlap, not a general gain. No fresh django turns exist to confirm directly
  (django instances finished before 20:58).
- SWE acceptance varies with content: run 3 scored 1.188 on the 19:24-19:55 SWE turns (early Python-10, incl. django)
  and 0.98 on later non-django turns. The decode-vs-prefill tap verdict stands (same content, both paths ~1.2-1.27).
- Consequences: the greedy gate on the same 3 django conversations (`accept_mt.py`) would be inflated the same way — not
  worth stopping the SWE run for. Decision metric from now: each kept checkpoint vs run 3 on recordings after its save
  time, split non-django SWE / chat / German (`fx-kolibri-keepckpt` keeps `out7/drafter-step<N>.pt` with the save time
  as mtime). The django held-out and the live held-out are debugging numbers only.
- Reading: ~1M recorded generated rows (a few hours) do not move the drafter on unseen data. Same scale where TandemLLM's
  drafter stalled (6.6M tokens); published drafters use ~100x more. Overnight data (Multilingual-105, German chat) is
  the test of whether more data generalises; a lower LR is a cheap side test, not the fix.
- Gate from now (agreed 2026-10-04 22:00): a checkpoint beats run 3 only with >= 5 % relative gain, consistently on
  several fresh slices (non-django SWE, chat, German), each scored on recordings after the checkpoint's save time.
  Clean general baseline: ~0.98 accepted per round on fresh non-django SWE, ~0.67 on chat (sampled path at temp 0.6,
  scored against Kolibri's top-1) — not the django-inflated 1.27-1.38 or the old ~65 % step-1 figure.
- Limit: later SWE turns carry no repo name (the reused prefix with the task is not re-recorded), so per-repo splits of
  fresh SWE are only possible by timing, as tonight's non-django one.
- Parked idea, separate from the general drafter: per-repo / per-session adaptation. Run 7 adapted to django fast, and an
  agent stays in one repo for hours; an online fine-tune per session could pay in serving.

## Run 8 (started 2026-10-04 22:15): from run 3, spread data (user: "yes, do we need a new start?")

- Why: run 7's rows were 65 % SWE from ~10-15 tasks (516k rows, ~35-50k per task) vs 35 % chat from 182 conversations
  (~1.5k each): the drafter learned a few repos deeply (django), nothing general.
- Architecture re-checked, unchanged: 1 layer (2 layers no gain, 4 slower; one layer does not even generalise yet),
  hidden 2560 / 20x128 heads (= Kolibri, shared head) / FFN 4096, taps 44/47/49 + token embedding (layer probe), depth 3
  (MoE verify cost), soft CE top-32 + head-state L1.
- New start from run 3 (run 7 = run 3 + django). Per-recording pass limits replace the global cap: SWE-agent requests
  once (`--passes-swe 1`), chat twice (`--passes 2`), counted per recording and kept across resumes (`uses` in the
  checkpoint). First pass: 467k SWE + 240k chat generated rows. `fx-kolibri-drafter-rec8` -> `out8/`, `train8.log`;
  `fx-kolibri-keepckpt` keeps `out8/drafter-step<N>.pt`.
- Start points (run 3): SWE django held-out 1.265, live 0.951, German held-out 0.494 (13 requests, 13,795 chains; all
  German recordings are clean for run 8, it starts from run 3).
- Streams: when x86 scores Python-10, `switch-ml105.sh` (x86) holds the old runner and starts Multilingual-105 with 3
  agent workers (`fullrun-kolibri-ml105.sh`); `chat3.sh` (GB10) then raises the mixed chat worker to 2 (chat 3 streams,
  SWE 3).
- Verdict as agreed: checkpoints vs run 3 on recordings after each save, >= 5 % on non-django SWE, chat and German.

## Run 9 (started 2026-10-04 23:05): scale via GLM-5.2 text through Kolibri's prefill (user: "do what you need")

- Why: NVIDIA's DFlash recipe (NeMo AutoModel) trains on Open-PerfectBlend prompts with target-regenerated responses,
  ~1.36M prompts; our self-generation gives ~10M tokens a day, ~1/500 of that. Kolibri's SFT teachers include GLM-5.2
  (tech report p53), and `mgoin/open-perfectblend-glm5.2-regen` (rev 003f54db, 1.42M conversations with reasoning,
  15.7 GB) is GLM-5.2's regeneration of all of Open-PerfectBlend. Prefill is ~10x cheaper per token than generating.
- Data: 5 of 32 shards (0, 9, 18, 26, 28; 2.75 GB, sha256 5/5 vs HF lfs.oid; `~/kolibri-drafter/glm/SOURCE.json`),
  `data_glm.py`: Kolibri's template with `preserve_thinking` (GLM writes `reasoning</think>answer`), default effort high
  (= serving's system preamble), whole conversations <= 16,384 tokens, shards round-robin: 14,265 conversations, 45.0M
  tokens. No German in PerfectBlend.
- Training (`run9.sh`, unit `fx-kolibri-drafter-run9`, `train9.log`, `out9/`): `train_mt.py` from run 3 (warm start onto
  taps 44/47/49, fuse identity on 49 — added to train_mt), 3-step rollouts, L1 + 0.1 CE (as run 3), lr 2e-4 constant,
  one pass, 29,673 steps at ~1,500 tok/s: ~8.3 h. Checkpoints kept every 500 steps (`fx-kolibri-keepckpt`).
- Window: Kolibri server, chat generation, run 8 and the x86 Multilingual run stopped 22:50 (ML105 had 12 minutes, 0
  finished; its resume keeps finished instances).
- After (`after9.sh`, unit `fx-kolibri-after9`): every 10th kept checkpoint + final vs run 3 on recordings run 9 never saw
  (fresh non-django SWE after 20:58, chat after 20:58, German chat), then the recording server, chat (2 mixed + 1 German
  worker) and ML105 (-w3) come back. Phase B (fine-tune the best run-9 checkpoint on Kolibri's own recordings incl.
  German) is decided on those numbers.
- Python-10 (Verified) score: 3/10 resolved, 5 completed (5 empty patches) — triage pending.

## Run 9 RESULT (2026-10-05 07:00): GLM-5.2 text at scale generalises — gate passed on SWE and chat

Fresh recordings run 9 never saw (`after9.log`; same chains for every checkpoint):

| fresh slice | run 3 | step 5000 | 15000 | 25000 | final 29,669 | gain |
|---|---|---|---|---|---|---|
| SWE non-django after 20:58 (218,608 chains) | 0.993 | 0.904 | 0.992 | 1.072 | 1.132 | +14.0 % |
| chat after 20:58 (185,147) | 0.601 | 0.603 | 0.691 | 0.735 | 0.763 | +26.9 % |
| German chat, all (178,754) | 0.502 | 0.423 | 0.468 | 0.498 | 0.509 | +1.4 % |

- Rising to the last checkpoint on SWE and chat, no plateau: 5 of 32 shards used; more GLM data is the next lever.
- German flat (PerfectBlend has none): phase B on Kolibri's own German recordings.
- The in-process SWE held-out (3 django conversations, train_mt's metric) fell 1.10 (step 1000) -> ~0.9: run 3's
  django skill fades; irrelevant to the fresh-slice verdict.
- Phase B started 07:15 as run 10: `train_rec.py --init drafter-run9-final.pt --lr 1e-4 --passes 2 --passes-swe 1`
  (unit `fx-kolibri-drafter-rec10`, `train10.log`, `out10/`, checkpoints kept). Judged the same way: kept checkpoints vs
  run 9 final on recordings after each save; it must lift German without giving back SWE/chat.

## Next data (2026-10-05 08:10)

- All 32 shards of `mgoin/open-perfectblend-glm5.2-regen` (rev 003f54db) on disk, sha256 32/32 vs HF lfs.oid
  (`~/kolibri-drafter/glm/SOURCE.json`). Shards 0-26 are `open-perfectblend` (math/code/chat/IF mix), 27 mixed, 28-31
  `ultrachat` (long multi-turn).
- `glm2` = shards 1-8 + 29 (same 4:1-ish mix as run 9's, no overlap): 21,483 conversations, 60.0M tokens (~11 h of
  training at ~1,500 tok/s). Unused: 10-17, 19-25, 27, 30, 31.
- Run 9 curves (fresh slices still rising at 45M tokens; training-stream top-1 flattening 0.677 -> 0.671 over the
  last 5k steps) say new data beats more passes. Next window: run 11 = run 9 final + glm2, then the same fresh check.

## German (2026-10-05 08:20, user: "is there enough german text? schedule like you need")

- Not enough Kolibri-written German: 143 recorded German requests, 184k generated rows (15.4 % of recordings) vs 45M GLM
  tokens. Chat generation shifted to German: German worker 1 -> 2, mixed 2 -> 1 (SWE keeps 3 streams).
- German text through the prefill path: `data_text.py` (any data.py shape, `SOURCE:SKIP`, Kolibri template at effort
  none: these answers have no reasoning) -> `glm/de`: sharegpt-deutsch from row 2,000 + alpaca-gpt4-de from row
  10,000 (gen_chat walks the same files from the start, so Kolibri's own German prompts and the German held-out stay
  out): 44,068 conversations, 12.8M tokens (sharegpt 4.6M, alpaca 8.2M). With glm2 (60M): German ~18 % of run 11.
- Tonight (22:00, timer): server down; `tools/kolibri/verifycost.py` (decode-path forward with 1/2/3/4/6 rows at
  1k/8k/32k context on real text; draft head full vocab vs 32k slice); run 11 = glm2 + de through prefill (~13 h);
  fresh scoring; restore. Start checkpoint (run 9 final or run 10) from run 10's fresh check.

## Run 10 RESULT (2026-10-05 08:21): phase B passes on all three fresh slices

Run 10 step 30000 (saved 07:16) vs run 9 final, recordings after 07:17:

| fresh slice | run 9 final | run 10 step 30000 | gain |
|---|---|---|---|
| SWE (100,029 chains) | 0.968 | 1.216 | +25.6 % (partly inflated: 3 ML105 instances in flight at 07:16 had their first turns trained) |
| chat, non-German (103,188) | 1.128 | 1.211 | +7.4 % (single-turn: clean) |
| German (34,346) | 0.745 | 0.847 | +13.8 % (single-turn: clean) |

- Run 11 (22:00 timer, `run11.sh` as unit `fx-kolibri-drafter-run11`, log `run11.log`) starts from run 10's newest kept
  checkpoint (`drafter-run10-window.pt`); verify cost first; scored after the restore on recordings made after it.
- Watchdog: `watchdog.sh` (root unit `fx-kolibri-watchdog`, every 5 min) writes `~/kolibri-drafter/STATUS` and
  appends `ALERTS` (units down for the phase, server health, new log errors, stalled training, disk/memory, x86 ML105);
  a 30-min heartbeat in the session reads it and fixes what it flags.

## Final test pool (reserved 2026-10-05 08:50; never trained on, never looked at until a final candidate)

The fresh slices have chosen between runs 9/10/11, so they are development data now. Untouched test pool:
- SWE: 9 whole repos (no sibling instances anywhere in training: the django lesson), 15 instances, 9 languages —
  jq (C) x2, laravel (PHP) x2, three.js (JS) x2, nlohmann/json (C++), prometheus (Go) x2, rxjava (Java), rubocop
  (Ruby) x2, bat (Rust) x2, valkey (C). x86 `/root/set-kolibri-test.txt`; removed from `set-kolibri-ml105.txt` (90
  left; original kept as `set-kolibri-ml105.orig.txt`). All sit at run-order index >= 55, beyond what ML105 reaches
  before tonight's stop; `run11.sh` logs a guard (any test instance started -> named in run11.log).
- German: sharegpt-deutsch rows 1,500-1,999 and alpaca-gpt4-de rows 8,000-9,999 (gen_chat walks from row 0 and is at
  ~200; data_text used 2,000+ / 10,000+). gen_chat reaches 1,500 in ~2 days at the current rate: cap it before then.
- Chat: ultrachat_200k shard 0 rows 60,000+ (gen_chat at ~350).
- Recorded once, at the end: server with `TENSORFOLD_KOLIBRI_RECORD=rec/test` (no trainer or dev check reads it), then
  one scoring of the final candidate vs run 3 / run 9.

## Surveys (2026-10-05): `notes/kolibri-drafter-survey.md`

- Shape: 1 layer, dense, late multi-tap is normal (DeepSeek's own post-hoc DSpark for V4.1: last 3 layers via 3d->d).
  Outlier: FFN 1.6x hidden vs 4-6.4x in EAGLE-3 drafters for MoE targets. Acceptance gap: peers 2.8-3.6 accept length
  at depth 3 (incl. bonus) vs ours 2.13 SWE / 1.76 chat (metrics/sampling differ; approximate).
- Training stories: regenerated / same-family cross-distilled data + a little self data; 3-4 epochs max; dense beats MoE
  drafters; data scaling far from saturated at our 45M (Meta ~96B, Together 20M->50M worth ~+10 % speedup); 5 taps >
  3; longer rollouts help code/math; LK loss lifted DeepSeek MTP 3.2 -> 4.8; gains shrink with concurrency.
- Lever order for us: more GLM data (run 11 -> ~118M) -> FFN ~10240 -> 5 taps -> longer SWE rollouts + LK loss for the
  own-data fine-tune; evaluate per language/domain at deployed concurrency.

## Training speed (2026-10-05 09:50)

- German batch was 44,068 conversations, median 186 tokens (92 % < 512): one Kolibri prefill + one step each, ~7 h
  instead of ~2.4 h. Packed to `glm/de4k` (3,713 sequences <= 4,096 tokens, same 12.8M tokens; conversations end to
  end, as data.py did for run 2's chat); run 11 uses it.
- Drafter-side step (2,048 rows x 3 steps, real head, GPU shared with the server so absolute ms inflated):
  full vocab chunked+recompute CE (current) 1,234 ms; full vocab plain CE 854 (1.4x); 32k vocab chunked 502 (2.5x);
  32k plain 446 (2.8x). `train_mt.py --plain-ce` (same loss: 16.416101 vs 16.416103, gradients bit-equal) is on for
  run 11. 32k draft vocab (frequency-selected, out-of-vocab rows L1-only) changes the loss: A/B in a later window.
- Further levers: overlap Kolibri prefill with the drafter step (streams), torch.compile the drafter, FP8 head logits,
  sparse anchor positions (only once the drafter side dominates).

## Loss / vocab A/B (window from 2026-10-05 09:54; user: "try aggressive optimizations first")

- Finding: train_mt used EAGLE-1's loss (state L1 + 0.1 CE on Kolibri's top token); EAGLE-3 credits dropping the
  state regression (token-level loss) for its data scaling. New `train_mt.py` options: `--loss l1ce|kl|kll1`,
  `--l1w`, `--draft-vocab` (losses over the slice; out-of-slice CE ignored), `--max-tokens`. `eval_rec.py
  --draft-vocab` scores the drafter as served (argmax inside the slice, mapped back to token ids).
- Draft vocab: top 32,768 tokens by frequency over Kolibri's own generated rows (before 2026-10-04 20:58, so the fresh
  slices stay out) + glm/glm2/de4k text: `~/kolibri-drafter/draft_vocab_32k.json`. Coverage of fresh Kolibri output
  (1.13M tokens): 16k 90.2 %, 24k 92.8 %, 32k 94.7 %, 48k 96.8 %.
- Arms from run 9 final, the same first 10M tokens of glm2 + de4k (same shuffle), lr 2e-4: A l1ce full vocab
  (control), B kl 32k (EAGLE-3), C kl + 0.1 L1 32k. (D l1ce 32k dropped on the user's call: only a diagnostic,
  expected worse.) Smoke speed: 32k arms ~2,100 tok/s. Scored full and 32k on the slices clean for run 9 (after
  2026-10-04 20:58). Runner `abrun2.sh` (unit `fx-kolibri-drafter-run11`, log `abrun2.log`); restores serving after.
- The 22:00 run-11 timer is cancelled; run 11 restarts with the winning recipe.
- DFlash/DSpark: plausible (z-lab's block-4 drafter for Qwen3.5-35B-A3B, a similar 3B-active MoE, reports accept length
  3.1-3.6) but 5-9x our params per token on a compute-bound box, and TandemLLM's 5-layer Kolibri drafter failed at
  6.6M tokens. Candidate arm E (block 4, 3-5 layers, 5 taps, from scratch) after this A/B.

## Width A/B: no effect -> capacity is not the limit (2026-10-06 14:20)

Run 13w = run 11 final with FFN grown 4096 -> 10240 (loss-free, 90.4M -> 137.6M) on run 12c's exact data and order;
run 12c is the narrow control. Fresh (after 09:55, 32k), 13w vs 12c at equal steps:

| step (tokens) | SWE | chat | German |
|---|---|---|---|
| 4,000 (6M) | 1.354 vs 1.327 (+2.0 %) | 1.229 vs 1.235 (-0.5 %) | 0.948 vs 0.950 |
| 8,000 (12M) | 1.330 vs 1.345 (-1.1 %) | 1.239 vs 1.237 | 0.948 vs 0.943 |
| 16,500 (25M) | 1.355 vs 1.371 (-1.2 %) | 1.248 vs 1.247 | 0.950 vs 0.947 |

All within noise, per step too. Stopped at step ~17,400 (user: "stop, then block drafter"). The 1-layer chain drafter is
not capacity-limited (dense or a layer-49 MoE copy would not help); run 12c's data-scaling also flattened (+1 % over 33M).
Next: block drafter (DFlash-style, parallel block, target features injected into every layer).

## DIRECTION (2026-10-06, user: "we concentrate on our own training and wait for zig engine")

TensorFold's Python engine is frozen (ashhart on #211, 2026-10-06): no Python PR merges, the native Zig engine
(`zig-preview`) is the future; Zig CUDA serves only Nemotron from the command line so far.
- Engine work on our fork stops: fused drafter step, adaptive depth, FP8 prefill, decode kernels, the drafter PR. The
  fork (`kolibri1-drafter`) stays as our measuring instrument (spec bench) and our own serving.
- Focus: drafter training. Data scaling with ~15-20 % German in every batch (run 12c, then glm4 ...), SWE kept by a
  share of agent text + a repo-disjoint SWE dev slice, later wider FFN / 48k vocab.
- Block drafter (arm E) gains weight: Zig already has one-pass drafting (`--block-lanes`) waiting for trained rows; a
  Kolibri block drafter would drop in once Kolibri runs on Zig.
- x86 SWE run stays paused.

## Run 12 stopped, run 12c instead (2026-10-06 05:40)

Run 12 (run 11 final + glm3, no German) vs run 11 final, recordings after 09:55, 32k: SWE 1.383 -> 1.329 / 1.349
(13M / 26M tokens), chat 1.230 -> 1.238 / 1.231, German 0.945 -> 0.825 / 0.773 (-18 %): without German in the mix the
drafter forgets German fast and gains nothing elsewhere. Stopped at 26M. Run 12c (`run12c.sh`, same unit name,
`train12c.log`, `out12c/`): run 11 final + glm3 + de4k (second pass of the German text), recipe C. Rule from this: every
GLM batch carries ~15-20 % German.

## ENGINE MEASURED (2026-10-06 01:42): roadmap gate 4 passed — ~1.23-1.26x at c=1, identical replies

`specbench.py`, run 11 final in TensorFold kolibri1-drafter (e72b31f), 32k slice, GPU idle, c=1, 256 tokens a prompt,
prompts nobody trained on (10 SWE turns from 5 ML105 trajectories finished 14:46-15:13, 13k-46k tokens; last 8 chat,
last 8 German); raw `notes/data/kolibri/specbench-run11.log`. Decode speed vs copy drafting alone:

| | SWE | chat | German | equal-weight mean |
|---|---|---|---|---|
| greedy d1 | 1.232x | 1.296x | 1.228x | 1.252x |
| greedy d2 | 1.218x | 1.338x | 1.221x | 1.259x |
| greedy d3 | 1.130x | 1.273x | 1.147x | 1.183x |
| served d1 | 1.263x | 1.255x | 1.163x | 1.227x |
| served d2 | 1.224x | 1.283x | 1.111x | 1.206x |
| served d3 | 1.142x | 1.197x | 1.012x | 1.117x |

- Copy-only 42.6-49.0 tok/s, learned 48-64 tok/s; learned drafts kept 55-63 % (d1), 39-49 % (d2), 24-39 % (d3).
  0 mismatches in every cell (greedy and served sampling). c=4 chat+German together: 110.0 -> 109.6 tok/s (neutral).
- Default: depth 1 (1.23x as served); d2 only pays on chat. Below the speed model's ~1.33x: sampled acceptance and the
  drafter's Python/launch overhead; next speed lever = CUDA graph / fused drafter step.
- Run 11 final fresh (vs run 10, after 09:55, 32k): SWE 1.486 -> 1.383 (axios-biased slice), chat 1.166 -> 1.230
  (+5.5 %), German 0.830 -> 0.945 (+13.9 %); raw `notes/data/kolibri/run11-fresh.log`. Run 12 started 01:42.

## Run 11 mid-run check (2026-10-05 22:10): chat/German up, SWE down on an overlap-biased slice

Kept checkpoints vs run 10 (start), recordings after 09:55 (clean for run 11), 32k as served:

| fresh slice | run 10 | run 11 @15M | @30M | @45M |
|---|---|---|---|---|
| SWE (229,977 chains) | 1.486 | 1.430 | 1.392 | 1.367 (-8 %) |
| chat non-German (38,240) | 1.166 | 1.169 | 1.200 | 1.219 (+4.5 %) |
| German (65,832) | 0.830 | 0.866 | 0.892 | 0.922 (+11 %) |

- The fresh SWE slice is axios + babel (ML105 after 14:27); run 10 trained until 09:54 on Kolibri's own recordings of
  earlier axios instances (4731, 5085, 5316 finished 08:58-09:31 local): run 10's SWE baseline is repo-biased (the django
  pattern). SWE dev scoring needs a repo-disjoint slice (repos absent from every checkpoint's training); the test pool has
  that property but stays reserved. If SWE still falls on a clean slice: mix SWE agent text back in (recordings once, or
  the Python-30 trajectories through prefill).
- Equal-weight mean ~+2.5 %, below the 5 % gate but rising; the in-training held-out (django) fell 1.51 -> 1.26.

## Engine integration (2026-10-05 evening, user: "yes put drafter into engine")

- TensorFold branch `jschmied/TensorFold:kolibri1-drafter` (on kolibri1-record): 82b9e0c `cuda/drafter.py` +
  decoder/engine wiring, e72b31f perf (preallocated entry buffer, fused qkv / gate-up, one host sync a chain).
  `TENSORFOLD_KOLIBRI_DRAFTER=<train_mt checkpoint>`, `_DRAFTER_VOCAB=<32k json>`, `_DRAFTER_DEPTH` (2),
  `_DRAFTER_STREAMS` (all). Copy drafts first; on a miss the learned chain computed at the end of the previous round;
  kept rows (and the prompt's last 2,048 rows) enter the drafter from the layers the engine taps anyway.
- Tests (tiny Kolibri): the chain equals an fp32 port of the training rollout (cos > 0.999, same first draft); replies
  with learned drafts equal serial (greedy and sampled, two streams together). Existing kolibri1 tests 16/16.
- `tools/kolibri/drafter/specbench.py`: copy vs learned depth 1/2/3 on prompts nobody trained on (5 SWE trajectories
  finished 14:46-15:13, last 8 chat / German), greedy and served sampling, identical-reply check, c=4 arm. Tiny dry run:
  0 mismatches; per-round drafter cost unmeasurable while run 11 holds the GPU at 95 % (11.8 / 16.5 / 20.8 ms at depth
  1/2/3 contended; bytes say ~2.5 ms at depth 2 idle). Runs idle between run 11 and run 12 (`run12b.sh`) ->
  `specbench-run11.log`.

## Verify cost MEASURED (2026-10-05 15:26) — roadmap gate 2 passed

`tools/kolibri/verifycost.py` on TensorFold kolibri1 (FP8, decode path, real SWE text), raw `notes/data/kolibri/verifycost-1005.log`:

| rows | 1k ctx | 8k | 32k | TandemLLM NVFP4 (for scale) |
|---|---|---|---|---|
| 1 | 21.36 ms | 22.15 ms | 25.12 ms | 12.5 ms |
| 2 | 1.154x | 1.131x | 1.175x | 1.32x |
| 3 | 1.268x | 1.258x | 1.283x | - |
| 4 | 1.388x | 1.372x | 1.405x | 1.86x |
| 6 | 1.718x | 1.615x | 1.703x | - |

Draft head per step: full vocab 3.62 ms, 32k slice 0.92 ms. Our slower single row (FP8, ~56 % of the byte ceiling)
makes extra rows relatively cheap.
Speed model, recipe C acceptance (32k scoring, chained acceptance split with equal per-step conditional rate), draft
step ~1.6 ms (head 0.92 + 90M layer read ~0.66), c=1, greedy: depth 1 SWE 1.37x / chat 1.33x / German 1.18x (mean
1.29x); depth 2 1.44 / 1.40 / 1.16 (mean 1.33x); depth 3 1.38 / 1.35 / 1.09 (1.27x). With 2x the draft cost, depth 2
mean ~1.22x. Gate (>= 1.15x on the equal-weight mix) passed -> engine integration is worth building. Open: sampled
acceptance (temp 0.6-1.0) is below top-1 match; concurrency unmeasured.

## Training focus (2026-10-05 15:25, user: "stop swe and concentrate on training")

x86 ML105 (15/90 done), chat generation and the server stopped; 22:00 timer cancelled; run 11 started 15:26
(`run11b.sh`: recipe C from run 10, glm2 + de4k, no restore after). Scored on recordings after 2026-10-05 09:55 (run 10's
last training): clean for run 11. Run 11: 25,196 conversations, 72.9M tokens, 48,248 steps at ~2,086 tok/s (ends
~01:15). `glm3` converted (shards 10-17 + 30: 25,479 conversations, 70.0M tokens). Run 12 chained (`run12.sh`, unit
`fx-kolibri-drafter-run12`): waits for run 11's scoring, then run 11 final + glm3 with recipe C (~9.5 h), scored with
run 10 and run 11 on the same clean recordings; server stays down. Watchdog now treats any `fx-kolibri-drafter-run1*`
unit as the window phase.

## Loss A/B RESULT (2026-10-05 14:27): KL wins -> recipe C (KL + 0.1 L1, 32k slice)

Arms from run 9 final, same 10M tokens (glm2 + de4k), scored on recordings clean for run 9 (after 2026-10-04 20:58);
raw log `notes/data/kolibri/abrun2-loss-ab.log`. Accepted per round of 3:

| slice (chains) | scoring | run 9 | A l1ce full | B kl 32k | C kl+0.1 L1 32k |
|---|---|---|---|---|---|
| SWE (644,916) | full | 1.159 | 1.211 | 1.266 | 1.272 |
| SWE | 32k (as served) | 1.125 | 1.167 | 1.225 | 1.232 |
| chat non-German (275,920) | full | 1.138 | 1.171 | 1.199 | 1.207 |
| chat | 32k | 1.114 | 1.143 | 1.168 | 1.176 |
| German (223,253) | full | 0.533 | 0.759 | 0.791 | 0.797 |
| German | 32k | 0.517 | 0.720 | 0.751 | 0.755 |

- KL (EAGLE-3) gains ~2x the current loss per token on SWE and chat (32k: SWE +9.5 % vs +3.8 %, chat +5.6 % vs +2.6 %)
  and trains 17 % faster (2,114-2,122 vs 1,813 tok/s). C >= B everywhere (small). German +39-46 % for all arms: the
  packed German text.
- 32k slice costs ~3 % vs unrestricted drafting (94.7 % coverage); 48k (96.8 %) is a later option.
- Run 11 (22:00 timer) uses recipe C from run 10's newest checkpoint; restore with SWE 2 workers, chat 2 + German 2.

## ROADMAP to a usable drafter (2026-10-05 14:30)

"Usable" = the learned drafter runs in TensorFold's kolibri1 engine, outputs identical with it on/off, measurably
faster on the mixed workload. **Weighting (user 14:35): Kolibri is weak at coding (SWE 20/30), so SWE counts no more
than text: SWE, chat and German weigh equally in every gate.**

1. Recipe (today): loss A/B (KL wins on full-vocab scoring); gate = B or C beats A on the mean of SWE/chat/German, also
   with 32k-slice scoring. Then fixed: KL (+0.1 L1), 32k vocab, 3 taps, 1 layer.
2. Speed possible (tonight, first minutes of the window): verifycost.py (1/2/3/4/6 rows at 1k/8k/32k; draft head
   full vs 32k) -> speed model (1 + accepted) / (verify + draft cost) per depth, on the equal-weight mix. Gate: >= 1.15x
   predicted at some depth; else the drafter stays a fallback behind copy drafting and decode kernels (TandemLLM's PDL /
   prefetch / fusion) become the lever.
3. Acceptance (nights 1-2): run 11 (recipe C from run 10, glm2 + de4k, ~73M tokens); own-recordings fine-tune with KL;
   next GLM batch if still rising; optional arm E (DFlash block 4). Gate per step: >= +5 % on the mean of the three
   fresh slices, no slice down. Final: once on the untouched test pool.
4. Engine (days 2-3): drafter forward in kolibri1 (fuse 44/47/49, own KV per stream, 32k head), copy drafts first,
   learned chain on a miss at the depth the speed model picks, existing multi-row verify, drafter KV refreshed from real
   target states; tests: identical outputs (greedy/sampled, solo/concurrent), engine drafter == Python reference, depth
   -> 0 at high concurrency. Gate: mean of SWE/chat/German >= 1.15x at c=1, none < 1.0x, no loss at c=6.
5. Ship (days 4-5): TensorFold PR (separate from #328); drafter weights on HF only with the user's go; afterwards
   periodic fine-tunes from recordings, retrain on Kolibri weight/template changes.
- Streams from tonight's restore: SWE 2 (`/root/fullrun-kolibri-ml105-w2.sh`), chat mixed 2 + German 2.

## Plan

Decision gate for run 5 (around 16:30-17:00, about 10M tokens seen): keep the multi-layer design if held-out step 1
clearly passes 65 % or accepted clearly passes 1.27, else fall back to a warm start (fuse passing layer 49 through,
layer 1 from run 3, new layers as identities: it cannot end below run 3).

Then, in order:
1. **Layer probe** (GPU free, minutes): every layer's state on the held-out conversations, top-1 for tokens t+1..t+3 by
   logit lens and a small linear probe. Pick 3-5 taps from data; candidates are the full-attention outputs
   (4, 9, ..., 49), since only those see past 513 tokens.
2. **Depth back to 1-2 layers** with the chosen taps: each drafter layer costs about 140 MB of reads per drafted token
   in serving, and depth did not help.
3. **Kolibri's own data (self-distillation)**: Kolibri writes SWE and chat replies overnight (server with copy
   drafting, about 50-120 tok/s), a few hundred thousand tokens a night; train on those contexts.
4. **Distribution loss**: KL to Kolibri's top-k probabilities instead of argmax CE (matters for sampled requests at
   temperature 0.6-1.0); drop the state regression once inputs are multi-layer, as EAGLE-3 does.
5. **Measure serving costs**: Kolibri's verify cost at 1-4 rows, drafter step with a 32k head slice in FP8; fit the
   cost model, choose chain depth (expected ~2-3) or a small tree.
6. **Engine integration** in `kolibri1` if (5) shows a clear gain over copy drafting: drafter when there is no copy
   match, copy drafts when there is; keep drafted == serial.
7. Later, if still short: small draft trees (4-6 rows), longer rollouts (5), DFlash-style block drafting.

Not on the list: changing Kolibri's weights; verification always stays exact.

## Where things are

- Code: `tools/kolibri/drafter/` (`model.py`, `model_mt.py`, `train.py`, `train_mt.py`, `accept.py`, `accept_mt.py`,
  `data.py`, `data_swe.py`, runners as `.sh.txt`).
- Data and checkpoints on the GB10: `~/kolibri-drafter/` (`swe.bin` 6.9M tokens, `chat.bin` 40M tokens,
  `drafter-*.pt`, logs). Chat sources pinned with sha256 in `~/kolibri-drafter/chat/SOURCE.json`.
- Results: `notes/data/kolibri/drafter1..3*/`; chart: https://claude.ai/artifact/MatxNcPBMPav889MRx2CpS
- TensorFold hooks (branch `kolibri1-cuda`): `forward(features=True)` pushed in #328; `taps=[...]` committed locally
  (f217fd3), not pushed.

## Run 14b — first block drafter (2026-10-06 14:40)

The chain drafter plateaued (run 11 / 12c / 13w at ~1.35 / 1.25 / 0.95, width null), so the next step is structural:
a DFlash-style block drafter (`tools/kolibri/drafter/model_block.py`). One pass drafts 4 tokens: the known token after
the anchor plus 3 mask embeddings; every one of its 4 layers (2560, FFN 6144, 372M) attends to the anchor's context
through keys/values projected from the fused taps 44/47/49, and to its own block bidirectionally. Trained from scratch
(`train_block.py`) on Kolibri's prefill: KL over the 32k draft vocab, weights 0.8^j, 512 anchors per 2,048-row window.
Data: glm + glm2 + glm3 + de4k ×3 = 213M tokens, 18 % German, at ~1,750 tok/s (≈34 h for all; checkpoints every 2,500
steps, bf16). Scored with `eval_block.py` on the same fresh recordings as run 11 (`accepted_3` is comparable with the
chain at depth 3; `accepted_4` is the block's own).

Hypothesis: position 0 reaches the chain's step-1 top-1 (~0.71 / 0.64 / 0.52) within ~40M tokens; later positions do
not compound their own errors, so `accepted_3` ends at or above run 11 and `accepted_4` adds 0.1–0.2. Position 0 below
0.5 on chat at 40M tokens means a bug (indexing or mask), not a capacity verdict.

**Last stage for any drafter: fine-tune on real serving recordings** (user, 2026-10-06). After the GLM pretraining,
the recordings (`rec/live`: Kolibri's own prompts and replies with taps 44/47/49 already stored) give the drafter the
real distribution. No prefill is needed, so this stage runs much faster than pretraining. For the block drafter it needs
a `train_rec`-style loop over `model_block` (todo). Rules carried over: pass caps per recording (run 6 memorised
88k rows), 15–20 % German, the reserved test pool left untouched, and scoring only on recordings made after the
checkpoint was saved.

**Run 14b → 14c → 14d (2026-10-06 afternoon).**
- 14b had no output-norm init (Kolibri's final norm scales states by ~43; train_mt copies it, train_block did not):
  KL stuck at ~8.4 after 5M tokens. Fixed and restarted 15:25.
- 14c (user): position weights decay^j easing 0.4 → 0.8 over 30M tokens (position 1 first), lr 6e-4 cosine to 6e-5
  over the 214M stream; resumed from 14b step 2,500 with the stream's first 3.77M tokens skipped (same order).
- 14d (user: check the later positions' conditional target): the target at row t+1+j is conditioned on the data's tokens
  t+2..t+1+j, but a draft at j only counts when Kolibri's own path took them. Kolibri's probability of GLM's prefix
  per position: 1 / 0.71 / 0.54 / 0.43 (in-training mean); on its own replies the greedy path holds for
  1 / 0.93 / 0.87 / 0.82 of anchors. 14d weights position j by that path probability (`--onpolicy`) and anchors only
  where t+2 is the assistant's (`--assistant-only`, as train_mt and the eval count). Resumed from 14c step 2,500.
- `test_block.py` (CPU): no leak from rows after the anchor, blocks isolated, solo == batched.

Fresh scoring (after 2026-10-05 09:55; accepted of 3, position-1 top-1):

| checkpoint | tokens | SWE | chat | German |
|---|---:|---|---|---|
| 14b step 2,500 | 3.8M | 0.22 / 0.20 | 0.35 / 0.27 | 0.28 / 0.21 |
| 14c step 2,500 | 7.5M | 0.27 / 0.23 | 0.41 / 0.31 | 0.34 / 0.25 |
| run 11 final (chain) | 73M | 1.38 | 1.23 | 0.95 |

recsync's prune (target 40 GB free, but only 34 GB of recordings left) began deleting the fresh scoring set at 17:00;
stopped after 65 old recordings (all PBS-verified). recsync now never prunes recordings after the scoring cut-off
(`--keep-after`), prunes below 45 GB and frees 10 GB at most.
- 14d fresh scoring (accepted of 3 SWE / chat / German; chat position 1): 11.3M 0.30 / 0.48 / 0.39, 0.35; 15.0M
  0.35 / 0.52 / 0.41, 0.37; 18.8M 0.37 / 0.54 / 0.43, 0.39. In training on GLM text position 1 is at 0.47-0.49: about
  0.1 of the gap is GLM vs Kolibri's own replies.
- 14e (user: yes, mix the recordings in now; tune as needed): resumed from 14d step 10,000 (22.6M tokens). Recordings
  mixed at 30 % of steps while they last: saved before 2026-10-05 09:55 (the scoring set stays out), not the live
  held-out, SWE 2 passes, chat 3 = 1,567 windows, 1.39M SWE + 0.38M chat rows. Teacher from the recorded layer-49
  state through Kolibri's final norm and head (= Kolibri's top-1 on 98.9 % of rows checked), so recordings train on
  the same full 32k KL as GLM, without a prefill. 1,024 anchors per window instead of 512: about 12-15 % slower
  (about 1,140 vs 1,300 tok/s with recordings in) for twice the training rows per prefill.

## Run 15 — block drafter built on run 11 (2026-10-06 19:45)

Diagnosis: the block's position 1 does the chain's step-1 job but was learning it from scratch on GLM text.
`model_block` `chain_ctx` + `from_chain`: context rows are the chain drafter's rows `fc(cat(embed(token r+1),
fuse(taps r)))`, block row 0 is the anchor's own row, rows 1-3 read a mask plus the anchor's fused state, causal inside
the block (row 0 sees exactly what the chain sees); fuse/fc/layer 0/out_norm copied from run 11 final (FFN padded with
zero-output columns), layers 1-3 identity at start. `test_block.py`: position 0 == chain step 1 (fp64, 5.5e-8: rope
runs in fp32; a 1 % change in one copied weight is caught). Untrained on fresh chat: accepted_3 0.667, top-1
0.637 / 0.047 / 0.029 / 0.018, above 14b-14e at 23M tokens (0.54). 14e stopped (its checkpoints kept).

German share while recordings are mixed: recordings are 8.1 % German (0.14M of 1.77M rows with passes), so about
15.0 % overall while they last (~5,500 steps), 18 % after. Watch the German slice; stratify if it drops.

Loss A/B (`run15ab.sh.txt`, auto-started when run 15 keeps step 2,500): from run 15 step 2,500, 1.5M GLM tokens each,
same data, fresh AdamW in every arm: A KL x 0.8^j x path (control), B KL accept-until-fail (supervise j while the
drafter's own drafts 1..j-1 match Kolibri's argmax) x path, C CE on Kolibri's argmax with position weights
d E[accepted] / d a_j from running rates (my approximation of D-PACE's idea; checked against a numeric derivative).
Not plain CE on data tokens: the data is GLM's text, not Kolibri's. train_block now also saves `resume.pt` (weights,
AdamW, step, data position, rng, recording windows left) for real resumes, and `--accum` for larger effective batches.

**Loss A/B result (21:32): a tie.** Fresh accepted_3 SWE / chat / German from run 15 step 2,500 (1.095 / 0.786 / 0.645):
A KL 1.109 / 0.808 / 0.655, B accept-until-fail 1.120 / 0.808 / 0.653, C acceptance-weighted CE 1.124 / 0.802 / 0.641
(`notes/data/kolibri/run15ab.log`). All within +-0.015: KL stays.
**Run 15 wore position 1 down:** chat position 1 0.637 (= run 11) at start -> 0.570 at step 2,500 (SWE 0.665, German
0.478), out of the >= 0.60 range; positions 2-4 learned (chat 0.67 -> 0.79 accepted_3), but through the layers position
1 shares. Still below run 11 on every slice (1.38 / 1.23 / 0.95).
**Run 15b (22:00): position 1 frozen as run 11.** `row0_chain`: block row 0's output is taken after layer 0; fuse, fc,
layer 0 and out_norm reloaded from run 11 and frozen; layers 1-3 and the mask train positions 2-4 only (position 1 out
of the loss). `test_block.py`: position 0 stays equal to the chain however layers 1.. move (without row0_chain the
same move shifts it by 1.12). Init from arm A with the chain reloaded: chat position 1 0.6366 exactly, positions 2-4
0.079 / 0.069 / 0.049 (they re-adapt to the restored layer 0). Recordings passes SWE 1, chat 2 (windows replay).
**recsync again:** the A/B's resume files pushed disk under 45 GB and the 21:35 prune freed 10 GB of the TRAINING
recordings (pre-cutoff; the scoring set is protected): 864 -> 411. Restored the 453 from PBS (rsync, 2026-10-05
08:48-09:36), prune threshold now 30 GB; resume files of finished arms deleted.
- 15b step 2,500 (7.0M tokens): fresh accepted_3 SWE 1.199 / chat 0.912 / German 0.729 (arm A 1.109 / 0.808 /
  0.655); chat top-1 0.637 (frozen) / 0.265 / 0.156 / 0.105, back above arm A's later positions. Run 11 chain still
  ahead (1.38 / 1.23 / 0.95). Recordings nearly used up at this point (41 windows left).
- 15b step 5,000 (≈9.5M tokens, GLM only after the recordings ran out): 1.217 / 0.941 / 0.749 (+0.02-0.03 per 2,500
  steps); chat 0.637 / 0.290 / 0.175 / 0.113.
- 15b step 7,500 (≈14M tokens): 1.207 / 0.966 / 0.770. SWE dips (-0.01) since the recordings ran out (SWE agent text
  only came from them); chat and German keep rising +0.02 per 2,500 steps.
- 15b step 12,500 (≈22M tokens): 1.203 / 0.998 / 0.786; chat 0.637 / 0.334 / 0.204 / 0.133. Chat and German +0.03 /
  +0.02 per 5,000 steps; SWE flat without agent text.
- 15b step 17,500 (≈29M tokens): 1.176 / 1.017 / 0.799. **SWE falling** (1.217 at 5,000 -> 1.203 -> 1.176): the GLM
  text has no agent sessions. Chat 0.637 / 0.347 / 0.214 / 0.139.
- 15c (02:55): first FULL resume (resume.pt: weights, AdamW, step, data position, rng, acceptance EMA) of 15b at
  17,500, continuing at lr 3.83e-4 without a re-warmup, recordings back at 10 % of steps for one more pass
  (`--rec-refresh`, SWE 1 / chat 1, 736 windows ≈ 7,000 GLM steps).
- 2026-10-07 03:25 disk at 40 GB: deleted out12 + out13w (33 GB, superseded chain runs) after a sha256 match against PBS (41 + 37 files identical), and the redundant resume-step17500.pt copy.
- 15c step 20,000 (≈33M tokens, 2,500 steps after the recordings came back at 10 %): **SWE 1.290** (from 1.176;
  accepted_4 1.411, above run 11's depth-3 1.38), chat 1.020, German 0.804; SWE top-1 0.714 / 0.465 / 0.317 / 0.229.
  Caveat: the fresh SWE slice is repo-biased (axios) and may share repos with the pre-cutoff recordings; a
  repo-disjoint slice is still open.
- 15c step 22,500 (≈36M tokens): 1.291 / 1.031 / 0.812 (accepted_4 1.410 / 1.077 / 0.849); SWE holds, chat/German +0.01.
- 15c steps 25,000 / 27,500 (≈40M / 44M tokens): SWE 1.304 / 1.300, chat 1.032 / 1.038, German 0.816 / 0.816.
  **Plateau**: chat +0.006, German flat, SWE holding after the recordings ran out (~25,800). Block (accepted_3) vs run
  11 chain: SWE 1.30 vs 1.38, chat 1.04 vs 1.23, German 0.82 vs 0.95; with all 4 drafts SWE 1.42 vs 1.38.

## Generation batch 2 + run 16 (2026-10-07 06:00, user: "yes and predecessor")
- Training paused at 15c step 27,500 (resume.pt kept). Recording server -> `rec/train2` (not rec/live: the scoring set
  stays fixed), chat 2 + German 2 workers (`gen_chat --max-rows 1400`: never reads the test-pool rows), SWE ML105 on
  x86 (-w3). `gen2.sh.txt`.
- Predecessor head (DSpark-style): `model_block.predecessor`, rank 256 (2.0M), identity at start; position j>=1's
  state + U silu(A h + B embed(token before)); trained on Kolibri's argmax as the predecessor (the accepted path),
  scored sequentially on the drafter's own previous draft. Identity check: 15c step 27,500 + head = chat 1.0384,
  exactly 15c.
- `--rec-all DIR`: recording dirs used whole (rec/train2), next to rec/live's pre-cutoff ones (checked: 0 scoring
  recordings selected).
- Run 16 (`run16.sh.txt`) waits for the generation workers (8 h), stops server and ML105, then trains from 15c 27,500
  with the head and recordings at 30 %.
- 06:20 disk: rec/train2 grew 11 GB in 16 min. SWE agent turns miss the prompt cache in this server session (cached
  0), so every step re-records its whole conversation (12-24K rows, ~150 MB); SWE was 10.8 of 11.4 GB. `rectrim.py`
  (loop, finished recordings only, .tmp + rename) keeps 1,024 rows before the first generated row and everything after
  (checked byte-exact on a copy: 16,455 -> 2,555 rows at keep 2,048, all generated rows kept, loads normally).
  Recording windows now end at the recording's end and step back (`rec_windows`), so a reply gets its context; 99.5 %
  of generated rows are covered as first drafts. If disk still falls under 45 GB, generation stops early (run 16
  starts on its own).
- 06:55 disk: still ~13 GB/h (SWE 7x the disk of chat per reply token). out11 (34 GB, run 11 step checkpoints) deleted after a sha256 match against PBS (99 files; its final == drafter-run11-final.pt, kept); rectrim keep 1,024 -> 512 for new recordings.
- 07:40 prompt-cache misses on SWE turns: our `decoder._resume` resumes from a FREE slot that kept the prompt, else
  the oldest; 7 clients (3 SWE agents + 2 chat + 2 German) on 6 slots kept the queue full, so a chat request took the
  slot an agent kept while it ran a tool (related symptom: TensorFold #459, Python Flash Next, no fix there). Chat and
  German down to 1 worker each: 2 of the next 6 SWE turns hit (72K of 74K cached, ~2K rows recorded); the misses
  were 109-111K prompts. Server restarted with --parallel 8 (8 x 131072, 25.4 GiB of caches, 14 GB available after;
  3.2 GiB a slot, 10 would leave ~7 GB), ML105 stopped around the restart and resumed.
- 08:20 (user: "use for training what fits and balance carefully"): PBS holds 1,656 pre-cutoff recordings pruned
  locally on 10-06 and unused since (3.0M rows, 1.19M generated = 3.5x the 0.34M in use; 1,324 SWE-size, recorded
  10-04 18:04 - 10-05 08:48). Restored into `rec/old` (never rec/live: the trimmer must not touch the scoring set),
  400 at a time, each chunk trimmed to keep 512 (`restore_old.sh.txt`); first chunk 400 recordings = 7.7 GB.
- Balanced recordings (`train_block.pick`, `--rec-mix`, `--rec-cap`): class by text (SWE / German by train_rec's
  word test / English chat), steps drawn from the class furthest below its share, recordings stop when a mixed class
  runs out (simulated: 0.500 / 0.299 / 0.201 to the end). Run 16b (replaces queued 16, `run16b.sh.txt`): recordings
  40 % of steps, SWE 0.5 / chat 0.3 / German 0.2, passes SWE 1 / chat 3 / German 3 -> overall German ~19 %, SWE ~20 %
  of steps; waits for generation and the restore.
- 08:50 disk for generation (rec/train2 ~11 GB/h with 8 slots): out9 + out12c (38 GB, chain runs; finals kept as drafter-run9-final.pt) and the 32 GLM parquet shards (15 GB; the packed .bin/.idx stay) deleted after sha256 matches against PBS.
- **Run 16b step 2,500 (14:45; predecessor head + all recordings, balanced): SWE 1.430 / chat 1.064 / German 0.839**
  (accepted_4 1.599 / 1.120 / 0.884) vs 15c 1.300 / 1.038 / 0.816. **First slice where the block beats run 11's chain
  (SWE 1.38).** SWE top-1 0.714 / 0.547 / 0.383 / 0.281 (position 2 +0.07). Generation batch 2 ended at 14:05 with
  3,559 recordings; 16b's pool 9,080 windows (SWE 4,211 / chat 3,612 / German 1,257), mix holding at 0.5/0.3/0.2.
- LR / cool-down A/B (`run16ab.sh.txt`, 14:53) from 16b step 2,500: A 4e-4 cosine (control), B 2e-4, C 8e-4, D
  cool-down 4e-4 -> 0 over 1,700 steps; 1.5M GLM tokens + recordings each. 16b's resume.pt kept as resume-step2500.pt.

## Kolibri-1 on the Zig engine (started 2026-10-07, user: "start like this, first without drafter")
Branch `jschmied/TensorFold:kolibri1-zig` (worktree ~/git/tf-kolibri-zig, from 1.0.1 041d14a).
- 5b6fe4c `families/kolibri1/config.zig`: config.json read and refused as the Python engine does; host tests (a
  broken shape check fails them).
- FP8 is general infrastructure (block FP8 128 x 128 is the official FP8 format; Flash Next's NVFP4 exports carry
  FP8_PB_WO layers too) and Zig CUDA has none: build it model-free first. Our Python FP8 kernels are plain CUDA, so the
  Zig path is the same device code: `zig/kernels/cuda/fp8_lane.cu` = qmmf.cu lines 14-298 without comments/ATen,
  FP8G instances BM 16/32/64 x cluster, fused, reduce (all 8 build for sm_121a). Next: Zig launcher with the Python
  host's choices (split_k by shape, bucket 16/32/64, fused from 256 rows, cluster slices on sm_90+), the
  `_fragment_order` repack, a GPU test against bytes captured from the Python kernel; then fp8/experts.cu.
- **LR / cool-down A/B (16:29, `notes/data/kolibri/run16ab.log`)**, fresh accepted_3 SWE / chat / German from 16b step
  2,500 (1.430 / 1.064 / 0.839): A 4e-4 1.442 / 1.067 / 0.848; B 2e-4 1.452 / 1.080 / 0.854; C 8e-4 1.393 / 1.051 /
  0.832; D cool-down to 0 1.463 / 1.082 / 0.856. Higher lr hurts, lower helps a little, the cool-down adds ~0.02 (under
  the 0.05 bar: the plateau is data/capacity, not lr noise). Run 16c (16:55): from arm B, 2e-4 cosine to 2e-5; a
  cool-down saved for the very end.
- Zig FP8 lane: GPU oracle 12/12 byte-identical (`notes/data/kolibri/fp8-lane-receipt.txt`); posted as TF #482 with
  claim issue #481.
- NVFP4 (user: "nvfp4 will be used for sure"): branch `jschmied/TensorFold:cuda-nvfp4` (bf8d716, stacked on #482's
  36549fe, not posted). `qmmf.cu` replaces fp8_lane.cu (same qmmf.cu device code, FP4 instances beside FP8G),
  `prompt16.cu` = nvfp4/prompt.cu's FP4 tile-4 prompt GEMM; `qmmf.zig` = the lane matmul for both modes, `fp8.zig` /
  `nvfp4.zig` = the repacks (qmm.pack words + Fp4Linear scale tiles); `tf-cuda-test nvfp4` checks repack, lane rows
  (1/3/17/40/300) and prompt rows (17/300/1000) against Python bytes. Host tests pass; GPU oracle queued for a pause at
  run 16c step 2,500 (`gpupause.sh.txt`), then the run resumes whole as 16d. EXL3: not now (no Zig reader; ashhart
  plans to take Yuepixel's #444 diff when an EXL3 backend exists).
- 17:25 GPU pause (23 s): FP8 on the renamed qmmf.cu 12/12 PASS again, NVFP4 18/18 PASS (repack + lane 1/3/17/40/300 +
  prompt GEMM 17/300/1000, [5120, 6144] and [12288, 5120]), all byte-identical to the Python kernels
  (`notes/data/kolibri/qmmf-nvfp4-receipt.txt`). Training resumed whole as 16d from step 2,500.
- 16c step 2,500 (≈48.9M): 1.463 / 1.092 / 0.858; 16d step 5,000 (≈51M): 1.468 / 1.095 / 0.864 (accepted_4 1.660 / 1.163 / 0.915). Slow gains; balanced recordings 6,822 windows left.
- FP8 experts (PR 2 of #481's plan): branch `jschmied/TensorFold:cuda-fp8-experts` (6ab43be, stacked on #482, not
  posted). `grouped.zig` = experts.route's plan (one block to 1,024 pairs, else rank / offsets / scatter);
  `fp8_experts.cu` = fp8/experts.cu's device code, six instances; `fp8_experts.zig` = packing, kernel and grid as the
  Python host. 18:37 GPU pause at 16d step 7,500 (6 s): packing + plan + gate-up + down byte-identical for decode 1 / 3
  rows and prompt 300 rows (2,100 pairs, wide plan), E 33, [512, 2560] (`notes/data/kolibri/fp8-experts-receipt.txt`,
  runner `gpupause2.sh.txt`). Training resumed from resume.pt.
- 19:12 GPU pause at 16d step 10,000: NVFP4 experts (`jschmied/TensorFold:cuda-nvfp4-experts`, cd4ee05, stacked on
  cuda-fp8-experts; nvfp4/experts.cu's three decode instances, prompt items of 16) 4/4 byte-identical, E 33 [512,
  2560], decode 1 / 3 rows and prompt 300 rows. Kolibri host side (`kolibri1-weights`, stacked on cuda-fp8-experts):
  core safetensors reads F8_E4M3 (ada7ea3); `names.zig` maps the released checkpoint's 116,303 tensors, none left
  over (d9579da); `pack.zig` packs layer 0 byte-identical to weights.load (8a7965b; oracle kolibri_layer.py).
  Receipt `notes/data/kolibri/nvfp4-experts-layer0-receipt.txt`, runner `gpupause4.sh.txt`.
- 19:48 run 16d stopped at its step 12,500 save (resume.pt) for generation batch 3 (user: "do generation batch"):
  `gen3.sh.txt`, chat 2 + German 2 workers for 12 h into rec/train2, SWE ML105 on x86. Before it, the Zig device
  loader (`kolibri1-weights` ca43e63, weights.zig) read layer 0's 15 device arrays back equal to weights.load's
  (`notes/data/kolibri/kolibri-device-layer0-receipt.txt`).
- 20:35 disk for batch 3 (~8 GB/h; recsync frees only old rec/live): `diskguard.sh.txt` stops the generators below
  15 GB free. User "yes": local rec/old (27 GB), out15b, out16b deleted after all 11,609 files matched PBS by sha256
  (`/mnt/bulk/gb10/kolibri-drafter`; recsync's rsync has no --delete). 87 GB free. **Restore rec/old from PBS before
  run 16d resumes** (its --rec-all reads it; `restore_old.sh.txt`).
- 20:55 batch 3's server started with --parallel 6 (gen2.sh never got batch 2's 07:40 fix): 7 clients on 6 slots,
  prompt-cache hits 135 of 2.82M tokens in 40 min, ~90k-token prefills per SWE turn. Restarted with --parallel 8:
  4.00M of 4.22M prompt tokens cached (95 %) over the next 8 min; gen2/gen3 runners now say 8.

## External review 2026-10-07 23:00: verdict and the morning rule

- Checked: DSpine (arXiv 2609.36173, 2026-09-28: gated adjacent injection at every drafter layer, Qwen3-8B mean
  acceptance length 3.77 -> 4.82 vs DFlash, +23.3 % SGLang throughput) and PCTree (arXiv 2608.02123, 2026-08-03: trees
  from a DSpark predecessor head without retraining, +3.1..29.5 %) exist as cited. Loss tuning stays closed (KL / AUF /
  CE tied).
- Correction to the review's "teacher -> self predecessor schedule": under greedy acceptance a drafted predecessor is
  accepted only if it equals Kolibri's argmax, so on every surviving prefix the self and the teacher (argmax)
  predecessor are the same token; the schedule changes nothing measurable at greedy. The real inconsistency is
  elsewhere: the target at j is Kolibri's distribution after the DATA token at j-1 (recordings are sampled), but the
  head is told the argmax. `train_block --pred-src data` feeds the data token (differs only on sampled rows).
- Built (CPU tests 4/4): `model_block` `spine_rank` = DSpine-style low-rank adjacent injection before layers 1..
  (row j >= 1 reads row j-1's state, RMSNorm -> A -> silu -> U, U zero so a loaded drafter keeps its function
  exactly; row 0 untouched, so row0_chain stays exact); `train_block --spine-rank`, `load(..., spine_rank=)`.
- Morning (`morning16.sh.txt`): when batch 3's generators end, stop the server and ML105, score 16d saves 5,000 /
  7,500 / 10,000 / 12,500 on the fixed slices. Hypothesis: 12,500 vs 5,000 +0.00..+0.03 on chat and German.
  Rule: >= +0.02 on both -> resume with recordings moved to chat/German (review: swe 0.3 / chat 0.4 / de 0.3; the
  German pool at cap 3 then runs out first, batch 3 adds to it); else A/B from 12,500: control, --spine-rank 256,
  --pred-src data. rec/old (27 GB, on PBS) is restored only for training, and only if the disk allows (its restore
  stops below 50 GB free).
- Later: PCTree-style verify trees fit the GB10 (3 rows 1.26x, 4 rows 1.37x one row's cost at 8k); needs tree-masked
  verify in the Zig server, which is chain-only today.
- 2026-10-08 07:53 batch 3 ended: chat 923 / German 912 conversations in the files (batch 3: +389 / +505), train2
  8,397 recordings, 25 GB free (recorder floor reached ~06:45; recsync freed 10 GB of pre-cutoff rec/live at 05:00).
- Morning score (`notes/data/kolibri/morning16-scores.log`), accepted_3 SWE / chat / German (accepted_4):
  16d 5,000 1.468 / 1.095 / 0.864 (1.660 / 1.163 / 0.915); 7,500 1.488 / 1.101 / 0.866; 10,000 1.478 / 1.105 / 0.870;
  12,500 **1.487 / 1.109 / 0.872** (1.690 / 1.182 / 0.927). Plateau as predicted (+0.019 / +0.014 / +0.008 over 7,500
  steps); gains sit at positions 2-4, position 1 is the frozen chain. Rule -> structure A/B, not a resume.
- 08:05 run 16e (`run16e.sh.txt`): from 12,500, fresh AdamW, 2e-4 -> 2e-5, 1.5M GLM tokens, recordings 40 % at swe
  0.3 / chat 0.4 / de 0.3 (batch 3 included, rec/old not restored): K control, S --spine-rank 256, P --pred-src data.
  Hypothesis: S or P >= +0.03 over K on chat or German; +-0.015 = tie.
- Arm U queued behind 16e (user "ok" to unfreezing position 1): `train_block --train-row0 0.1 --pos1-weight 2`
  trains position 0's chain path (fuse, fc, layer 0, out_norm) at 0.1 x lr in its own AdamW group, position 0 weighted
  2 (defaults keep the freeze, so 16e's S/P arms are unchanged). `run16u.sh.txt` via `q16u.sh.txt`; scored with
  12,500 and K. Hypothesis: position 1 top-1 +0.005 or more on chat/German, SWE not lower; else the freeze stays.
- Run 16e (09:18, `notes/data/kolibri/run16e.log`), 1,621 steps an arm, accepted_3 SWE / chat / German (accepted_4):
  12,500 1.487 / 1.109 / 0.872 (1.690 / 1.182 / 0.927); K control (mix 0.3/0.4/0.3, batch 3) 1.504 / 1.112 / 0.876
  (1.720 / 1.185 / 0.933); S spine 256 1.506 / 1.113 / 0.876 (1.724 / 1.185 / 0.932); P pred-src data 1.506 / 1.113 /
  0.876 (1.721 / 1.187 / 0.932). **All tie** (S, P within +-0.002 of K): hypothesis (>= +0.03) rejected at this
  budget. The spine did train but stayed small (U Frobenius ~5.1 per layer vs layer 1's o 175). K over 12,500: SWE
  +0.017, chat +0.003, German +0.005 - the chat/German mix shift alone moves little in 1,621 steps.
- **Arm U WINS** (09:46, `notes/data/kolibri/run16u.log`): position 0's chain path unfrozen at 0.1 x lr, weight 2,
  same 1,621 steps as K. accepted_3 SWE / chat / German: K 1.504 / 1.112 / 0.876 -> **U 1.618 / 1.141 / 0.916**
  (accepted_4 1.851 / 1.216 / 0.974); position 1 top-1 0.714 -> 0.764 / 0.637 -> 0.651 / 0.521 -> 0.547, positions
  2-4 also up. The freeze was the cap: run 15 lost position 1 at full lr with the rest; at 0.1 x lr in its own group it
  learns the new data. U now beats run 11's chain on SWE (1.38) and closes on German (0.945); chat still under (1.23).
- 09:50 run 16f (`run16f.sh.txt`): arm U continued over the GLM set, same recipe, saves every 2,500
  (`fx-kolibri-keepckpt16f`). Next A/B candidates from a 16f save: row0 lr 0.3 x; pos1 weight 4.
- 11:05 run 16g (`run16g.sh.txt`, user: "if dspine does not hurt but potentially helps, we should use it"): run 16f's
  step 5,000 + --spine-rank 256 (zero at start), same U recipe; 391.3M trainable. 16f's resume.pt dropped.
- 11:13 backup + cleanup (`backupclean.sh.txt`, log `notes/data/kolibri/backupclean-1008.log`, user: "protect generated
  chats (backup first)"): own_chat*.jsonl, generator logs, chat/ sources and rec/train2 (58,851 files) synced and
  sha256-verified on PBS, 0 mismatches, kept locally. 23 superseded runs/checkpoints deleted only after every file
  matched PBS (out16c steps <12,500 + resume, out16e-S/P, out16ab-*, out14*, out15, out10, out8, out7, out5, ab,
  drafter-mt4-2500.pt). 17 -> 50 GB free.
- 14:12 score of run 16g (`notes/data/kolibri/score16g.log`), accepted_3 SWE / chat / German (accepted_4): arm U 1.618
  / 1.141 / 0.916; 16g 5,000 1.683 / 1.178 / 0.957; 10,000 1.696 / 1.191 / 0.967; **12,500 1.710 / 1.200 / 0.973**
  (1.964 / 1.284 / 1.038). Position 1 0.800 / 0.673 / 0.576. **German now above run 11's chain (0.945)**, chat
  0.03 under it (1.23), SWE +0.33. Still rising (+0.013 / +0.009 / +0.006 per 2,500 steps). 16g resumed 14:16.
- 15:20 16g's recordings ran out at step 16,500 (German's 1,977 windows first; the trainer then drops recordings for
  every class) and it trained text-only after. Armed: at 16g step 17,500 score 12,500 / 15,000 / 17,500, then run 16h
  = second recording pass from 17,500 (`switch16h.sh.txt`, `run16h.sh.txt`).
- Generation cannot run beside training on the GB10 (two copies of the 77 GB target do not fit in 128 GB); they
  alternate. German caps the recording phase (~2,000 windows at share 0.3; SWE 5,575 and chat 5,541 have room), so
  batch 4 (`gen4.sh.txt`, user: "yes, generate as needed") is German only: 6 workers, sharegpt-deutsch < 1,400 and
  alpaca-gpt4-de < 7,900 (rows 1,400-7,900 never read: clean of the scoring set, whose prompts sit in own_chat*.jsonl
  and are skipped, of the test pool 1,500+ / 8,000+, and of de4k 2,000+ / 10,000+). Starts at 16h's step 12,500
  save (~19:00, `switchgen4.sh.txt`), 12 h.
- Drafter for generation? No: the Python server's drafter slot loads only chain (train_mt) checkpoints, so not the
  block drafter, and at c=4 run 11's chain was neutral (110.0 -> 109.6 tok/s, 2026-10-06); batch 4 runs more streams.
  Instead batch 4 got 8 workers (3 sharegpt-de + 5 alpaca-de), one per server slot (edited before it started).
- 15:37 16g scores (`notes/data/kolibri/score16g-17500.log`), accepted_3: 15,000 1.704 / 1.203 / 0.977; **17,500
  1.716 / 1.205 / 0.980** (accepted_4 1.972 / 1.291 / 1.046). The ~1,000 text-only steps did not hurt; gains are
  now small (+0.006 / +0.005 / +0.007 over 5,000 steps). 16h (second recording pass) started 15:40.
- **Draft vocab is a cap** (16:30): held-out top-1 inside the 32k slice: chat 0.970, SWE 0.944, **German 0.896** -
  a German chain of 4 is draftable at most ~0.64 of the time. Candidates by Kolibri's own top-1 counts over rec/train2
  (6.03M generated rows): train2 top 32k 0.957 / 0.979 / 0.933 (chat/SWE/German); current + train2 to 40k 0.977 /
  0.983 / 0.940; **to 48k 0.982 / 0.989 / 0.958**; to all seen (60,614) 0.987 / 0.993 / 0.975. Written:
  `draft_vocab_48k.json` (the 32k in order, then on-policy tokens; counts `top1_counts_train2.npy`). The head is
  Kolibri's frozen lm_head, so a wider slice needs no retraining to be scored: `switchgen4b.sh.txt` scores 16g 17,500
  and 16h 12,500 with 32k and 48k at the switch to batch 4 (~19:00). Draft head cost 32k 0.92 ms -> ~1.4 ms est.
- Vocab for 99 % held-out top-1 (order: the 32k, then train2 counts): SWE 52k, chat 76k, German 91k, pooled 68k,
  classes equal 78k; 95 % at 33k / 30k / 44k. Long sparse tail: the held-out itself reaches 99 % with 1.6k (SWE) /
  3.8k (chat) / 8.4k (German) distinct tokens, but many never occur in train2's 6.03M rows - more German data
  shrinks the needed vocab. Head cost ~linear in rows (32k 0.92 ms, full 3.62 ms). `draft_vocab_64k.json` = every
  token seen in train2 (60,614) added to tonight's scoring (`switchgen4c.sh.txt`: 32k / 48k / 64k).
- 18:49 (`notes/data/kolibri/vocab-ab-round2.log`), accepted_3 SWE / chat / German:
  | | 32k | 48k | 64k (60,614) |
  |---|---|---|---|
  | 16g 17,500 (round 1) | 1.716 / 1.205 / 0.980 | 1.787 / 1.214 / 1.007 | 1.793 / 1.219 / 1.016 |
  | 16h 12,500 (round 2) | 1.732 / 1.217 / 0.990 | 1.802 / 1.227 / 1.018 | **1.807 / 1.231 / 1.026** |
  Round 2 gains on held-out (+0.016 / +0.012 / +0.010 at 32k): not memorisation -> round 3 after batch 4. The wider
  vocab alone, no retraining: 64k +0.075 / +0.014 / +0.037 (accepted_4 2.077 / 1.322 / 1.096). **All three slices now
  above run 11's chain (1.38 / 1.23 / 0.945).** Next training trains on the 64k slice. Batch 4 (German) started 18:58.
- Licenses for a release (2026-10-08 19:30, user: "yes, check"; not legal advice): Kolibri-1 Apache-2.0. Prompt
  sources: ultrachat_200k MIT, sharegpt-deutsch Apache-2.0, self-oss-instruct ODC-BY (attribution),
  mayflowergmbh/alpaca-gpt4_de no tag = reformat of FreedomIntelligence/alpaca-gpt4-deutsch (Apache-2.0; content is a
  German translation of GPT-4 alpaca answers). GLM text: mgoin/open-perfectblend-glm5.2-regen "other" = prompts of
  mlabonne/open-perfectblend (Apache-2.0, its sources Apache/MIT) with answers by GLM-5.2 (MIT). gen_chat sends only
  the first user turn; every recorded answer is Kolibri's own. Only de4k (prefill text, sharegpt 2,000+ / alpaca
  10,000+) feeds third-party answers (ChatGPT / translated GPT-4) as context, Kolibri's distribution the target.
  -> Release under Apache-2.0 with an attribution list; batch 4 on alpaca-de prompts is fine.
- 19:20 batch 4 runs ~250 German conversations/h (batch 3: ~40) and ~6 GB/h of recordings; the recorder stops below
  25 GB. Dropped 16h steps 2,500-10,000, 16g 12,500/15,000 and out16c after PBS sha256 match (45 GB free).
  `floorswitch.sh.txt`: below 27 GB (or when batch 4 ends) stop generation and start round 3 = run 16i
  (`run16i.sh.txt`: from 16h step 12,500, 64k draft vocab, all recordings incl. batch 4). Its log line still says
  "16g" (copied text); paths are 16i.
- 19:59 (user: "backup and remove"): /opt/llm/models/qwen38-flash-next-mtpfp4-plebf16 (102 GB, no hardlinks) synced to
  PBS /mnt/bulk/gb10/models, 12/12 files sha256-matched, deleted locally; sums in
  /opt/llm/runners/SHA256SUMS-qwen38-flash-next-mtpfp4-plebf16.txt. fp8head kept (prod checkpoint; 120 GB of it
  hardlinked, ~12 GB would free). 137 GB free: batch 4 runs the night, floorswitch then starts run 16i at its end.
- 2026-10-09 07:00 batch 4 ended on time: 3,386 German conversations (sharegpt-de pool <1,400 used up at 05:25 after
  845; alpaca-de the rest), ~6 GB/h, 77 GB free at the end. Recording windows now swe 5,575 / chat 8,241 / **de
  10,275** (was 1,977; some alpaca-de chats classify as chat). At 0.3 / 0.4 / 0.3 SWE now binds a pass (~18.6k
  windows: chat 7.4k, de 5.6k of 10.3k used) - a later mix can lean to German/chat.
- 07:05 run 16i (round 3) started: from 16h step 12,500, 64k vocab, skip 92,985,802. (Its log line says "16g from
  16f step 5000" - copied text; INIT and skip are 16h's.)
