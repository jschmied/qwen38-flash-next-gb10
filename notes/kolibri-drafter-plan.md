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
