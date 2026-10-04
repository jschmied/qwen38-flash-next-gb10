# Kolibri-1 drafter: results so far and plan

Kolibri-1 (Aleph Alpha, 78B MoE, 3.5B active, FP8) ships no MTP head and no draft model exists for it. We serve it on
our TensorFold family `kolibri1` (PR #328) with copy drafting from the context. This note tracks the learned drafter.
Last updated 2026-10-04 14:50.

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
| 5 | 4 layers, MLP 8192, 389M, taps 12/25/49, from scratch | SWE + chat | 53.2 % (step 2,000) | 0.962 | running, catching up |

Findings:
- First-guess agreement stops at about 65 % whatever the size (runs 2-4): depth is not the limit.
- Rollout training lifts later steps (conditional step 2: 39 -> 45 %, step 3: 37 -> 44 %), well short of the 60-70 %
  in the EAGLE-3 / HASS papers.
- Greedy path (58 %) is below the text path (65 %): the drafter learned mostly other models' text.
- Estimated value of run 3 in serving, depth 2 with a 32k draft-head slice: about +25 % single-stream decode (verify
  cost per row not yet measured). Copy drafting gives +13 % (chat) to +31 % (greedy code).

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
