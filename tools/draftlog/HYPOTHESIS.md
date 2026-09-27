DRAFTLOG (two-candidate branching + TF1 confidence stop, offline replay), 2026-09-27, before the run. User: "is it possible
to verify two possible draft token on position n", "maybe if we have two draft token with similar probability?", "ok".
Run: clone venv rssm (PR code + dprob + boundary fix) with FNDRAFTLOG, prod config (K=5, probabilistic drafting over the
32k NVFP4 slice), greedy c=1, codeprobe's 4 code + 4 prose prompts x 700 tokens. Log per verify step: drafter top-2
(ids, probs) per position, fed tokens, target greedy token per position. Replay: tools/draftlog/replay.py.
Arithmetic before the run: a branch that succeeds gains exactly ONE token (the chain already emits the target's token at
the break); one extra verify row costs ~3.3 ms of a ~66 ms step (5 %).
H (branch, fixed extra row at the min-margin position): code -3 ... +1 %, prose -1 ... +4 %; only positive if the
  top-2 rescue rate at the break is >= ~50 % (prior guess ~35 %).
H (conditional branch, margin < t): the better variant if branching pays at all, but needs variable verify shapes.
H (TF1 confidence stop, p1 < tau): code ~0 (breaks are late), prose +2 ... +6 % (frequent early breaks); TensorFold saw
  +9 % on greedy code with a 3-4 ms verify row and no RecoverSSM.
Decision rule: build nothing below +3 % predicted; any engine work must decide on the GPU (no host sync, §5b).
