# TensorFold host-side cost: draft-chain syncs and the eager concurrent path (2026-10-01)

Question (tensorfold-opportunities "Speed-of-light agenda"): how much of TF 0.6.0's decode time is host-side —
(1) the per-draft-step round trip in `draft()` on the serial path, (2) eager launches on the multi-stream path.

Arms, EXL3 3.05bpw, greedy, 512 tokens of code after a 64-token warm-up, one process per arm, timing runs without
nsys, then one nsys run per arm for GPU duty cycle (busy-union / capture window) and syncs per token:
- S1 serial, graphs on (default); S0 serial, graphs off; M2 streams=2 at c=2; M4 streams=4 at c=4.

Hypotheses (ranges before measuring):
- S1 GPU duty cycle 75–90 %; host gap per draft step 0.1–0.4 ms. If duty ≥ 90 % the draft-chain lever is < 5 % and
  not worth a restructure; if ≤ 80 %, worth a design note.
- S0 vs S1: graphs worth 10–30 % ms/token on the serial path (TF launches many small kernels per layer).
- M2 / M4 GPU duty 40–75 %; aggregate tok/s M2 1.3–1.7× S1, M4 1.8–2.8× S1. A duty ≤ 70 % at c=2 says graphs for the
  concurrent path are worth pricing.
- Outputs: S1 and S0 greedy token hashes identical (TF's exactness contract); M2/M4 streams equal to solo per
  contract (same prompts) — a mismatch voids nothing here but is a finding.
Void: an arm whose log lacks its `ARM` json line, or whose engine reports a different streams/graphs setting.

## Round 2 (2026-10-01): the maintainer's lone-stream fix on the 0.6.1 branch

`origin/pr-141-0.6.1` = `cb5101d` adds `multi_solo.py` (a lone stream moves into a graph slot and replays the serial
graphs). Same probe, worktree `~/git/tf-061`, own extension cache. Arms S1, L4, M2, M4, two starts each.
- L4 on 0.6.1 = S1 on 0.6.1 within ±2 % (12.0–12.4 ms/token): the +10.9…+12.7 % gap closes. If L4 stays ≥ 13.3 the
  solo path is not taken (check why before anything else).
- S1 on 0.6.1 within ±2 % of 0.6.0's 12.09–12.19; M2/M4 within ±3 % of 0.6.0 (111.5–111.9 / 157.9–158.4 tok/s).
- Hashes: L4 = S1 on 0.6.1; equal to 0.6.0's `1bb116eb6ff5` unless 0.6.1 changed prompt handling (#163 keeps entries
  one token early — affects the kept point, not greedy tokens; a change is a note, not a void).

## Round 3 (2026-10-01): fix for the 0.6.1 solo-slot recapture

Round 2 result: 0.6.1 L4 18.41/18.56 ms/token (+35 % vs 0.6.0's eager 13.5), M2 99.8/99.1 (−11 %), M4 146.8/145.1
(−8 %), S1 unchanged. Cause measured (CAPTURE_LOG): `_grow`/`_shrink` call `_state_changed`, which drops the solo
slot's graphs; a lone request recaptures 23 graphs (3.02 s) when its slot grows past 256 rows; M2 8 (0.93 s). With
the slot grown during the warm-up (WARM_TOKENS=600): 0 captures, 12.13 ms/token = S1.
Fix (`~/git/tf-061-fix`, branch `solo-graph-keep`): the solo slot keeps its rows when idle (released only under
memory pressure), grows by doubling (≥ 8192), and `warm()` captures its graphs last, at the rows requests will find.
Arms S1, L4, M2, M4 x2, CAPTURE_LOG=1, default warm-up (64).
- captures in the measured run: 0 in every arm.
- L4 12.0–12.4 ms/token (= S1); M2 within ±3 % of 0.6.0 (111.5–111.9 tok/s), M4 within ±3 % (157.9–158.4).
- hashes unchanged (`1bb116eb6ff5` …). Out of range = the fix is incomplete; find the remaining resize first.

## Round 4 (2026-10-01): T7, our 32k draft vocabulary on TF 0.6.0

TF's default draft head scores 79,591 ids (`draft_vocab.txt`); ours (`tools/draft_vocab/draft_vocab_32768.txt`, det-135:
+6.4–6.8 % c=1 on vLLM vs the full 248k head) scores 32,768. Arms S1 and M4, default vs 32k, alternating, two rounds.
- S1 −1…−4 % ms/token with 32k; M4 −1…−3 %. Smaller than vLLM's gain: TF's default head is already 79k, not 248k.
- Accepted drafts within −1 pp of default (code prompts; §5k coverage 99.0–99.9 %).
- Hashes identical to default (drafts never change TF's output); a difference voids the run.

Round 4 result (code): per round −2.2…−2.7 % at c=1, c=4 +3.8…+4.5 % tok/s, total rounds 536 vs 532, hashes identical.

## Round 5 (2026-10-01): T7 on prose and other languages

PROMPT_SET=prose: English, German, Chinese, French prose, 512 tokens each. M4 (all four) and S1 (English), default vs
32k, alternating, two rounds. Our 32k list drops most ids below 65,536 (TF keeps all of them).
- English: rounds within ±3 %. German/French: rounds +0…+8 % with 32k. Chinese: rounds +5…+25 % (fewer CJK ids).
- If total M4 rounds rise > 5 %, the 32k list loses on prose despite the cheaper head: no proposal.
- Hashes identical to default in every arm.

## Round 6 (2026-10-01): draft depth x confidence on TF (#136 context)

TF default depth 6, τ 0.7. vLLM replay (§5ah, our cost model) predicts depth 7 + τ 0.8 at +10 % code / +12.8 % prose
over fixed depth 5; against TF's own depth-6/τ-0.7 stop the remaining gain should be small. S1 (one stream), code
and prose sets (S1 uses each set's first prompt only — so also an M4 pass over all four prompts, per-stream rounds and
per-round cost read, not aggregate tok/s). Cells: 6/0.7 (default), 7/0.8, 8/0.8, 6/0.8. Two rounds.
- 7/0.8 vs 6/0.7: code 0…+3 %, prose −1…+2 % (ms/token, S1).
- 8/0.8 within ±2 % of 7/0.8; 6/0.8 −1…+2 % vs default.
- Hashes identical across all cells (drafts never change output).

## Round 7 (2026-10-02): T12 gain bound — concurrent round forward eager vs one graph replay (`t12_bound.py`)

#180 head (`d2e651a`), EXL3, c=2 and c=4, 512 tokens a stream, code + prose prompts, every 15th full round measured:
the round's forward (and its first MTP step) captured as it stands, eager vs replay in 3 alternating blocks of 5.
§5at's traced idle (12 % at M2, 10.5 % at M4) bounds what graphs can remove.
- Forward: replay saves 0.8–2.5 ms of a ~25–45 ms forward at c=2, 0.6–2.5 ms at c=4; every block faster.
- MTP step: replay saves 0.3–1.0 ms a step (it is short and launch-heavy).
- Sum over a round (forward + mean MTP steps × MTP saving) = 6–11 % of the round wall at c=2, 5–10 % at c=4.
- Below 4 %: T12 is not worth the 2–3 days; above 12 %: the probe measures more than launch overhead (check the method).
- **Result (§5bc):** forward saved 3.9–4.05 ms (out of range high), MTP 0.14–0.21 ms a step (out of range low); round
  total 7.8 % (c=2), 5.6–6.2 % (c=4): in range.

## Round 8 (2026-10-02): #180 on an NVFP4 checkpoint (SvangenStudios saw no regression on cb5101d with NVFP4)

`mtpfp4` (NVFP4 experts, block-FP8 dense), cb5101d vs #180 (d2e651a), host_probe S1 / L4 (512 tokens, CAPTURE_LOG=1)
and solo_switch (four lone requests A, B, C, A'), two rounds.
- cb5101d L4 recaptures like EXL3 (captures_run 15–35) and runs ≥ 10 % slower than S1; #180 L4 has 0 captures and is
  level with S1. If cb5101d L4 has 0 captures on NVFP4, the regression is EXL3-specific and SvangenStudios' result
  is expected.
- **Result (round 8, §5bo):** as predicted — cb5101d L4 +31…+34 % over S1 with 21 recaptures; #180 L4 = S1, 0 captures;
  lone requests 26–29 captures each on cb5101d, 0 on #180; hashes identical.
