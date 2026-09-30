# FNHEAD4: NVFP4 W4A16 target lm_head (2026-09-30, user: "do quality run")

Overlay `patch_head4.py` + `fn_head4.py` (clone venv `vllm-venv-rssm`, env `FN_TARGET_HEAD_NVFP4=1`): the target's FP8
block-scaled lm_head is re-quantized at the first compute_logits to NVFP4 (per-16 E4M3 scales, one FP32 global) and
served by the draft head's Triton W4A16 GEMV. Unit test `test_head4.py` (real head): kernel = its own dequant (max rel
0.2 %); vs FP8 on proxy inputs KL 0.002 nats.

## `head4q` (before the run)

Clone venv, prod K5 config (HC fusion, F4 graphs, K5, NVFP4 draft head, KV 4 GiB), head4 vs FP8 head, 2 starts,
alternating. Probe `h4probe.py` per start: nvprobe (speed, hashes), `tfprobe.py` (teacher-forced NLL and top-1 on
reference text: 40 HumanEval solutions, 40 GSM8K answers), `evalprobe.py` (GSM8K 1319 + HumanEval 164, greedy, c=16;
scored by evalscore.py with McNemar).
- Speed: code c=1 −1…−2.5 % ms/tok (0.30 GB less per verify on a ~64 ms cycle), prose −1…−3 %, TTFT ±1 %; greedy
  hashes differ from base (the target distribution changes).
- Teacher-forced: top-1 accuracy Δ within −0.5 pp and NLL +0.3…+2 % vs FP8 on both sets. Out of range (top-1 worse
  than −1 pp or NLL > +3 %) = a real head loss.
- Tasks: GSM8K / HumanEval within start-to-start noise (every head4-vs-base McNemar p > 0.05, and head4-vs-base
  discordance no larger than base-vs-base).
Per the quality rule this is a screen, not a clearance: a pass makes it a candidate for a longer real-task run.

`head4q` was interrupted on 2026-09-30 07:0x (user: "interrupt work, do both now") after one complete head4 start;
`head4q2` repeats the full screen (2 starts per arm) on the clone venv now carrying the #58863 head (greedy text
identical to before, measured). Same hypotheses.
`head4q2` was interrupted too (2026-09-30 07:47, user: "interrupt, do tensorfold changes now") before any start completed; `head4q3` is the same screen.
`head4q3` was interrupted (2026-09-30 08:22, user: "do all now", TensorFold PR); `head4q4` repeats the screen from scratch.
