SWE (the deciding quality benchmark for the precision cuts), 2026-09-27, written before the run. User: "NVFP4 GDN need
reals becnhmark", "bf16 SSM state also need benchmark. we have to pick smallest quality trade".
Arms: fp8gdn (prod), nvfp4gdn, bf16ssm; exact prod config (K=5, dprob, RecoverSSM + boundary fix, block 1728) but
FN_MAXLEN 65536 and default KV (the smoke run hit ContextWindowExceeded at 32k with max_tokens 16000). 2 starts each.
Agent: mini-swe-agent 2.4.5 on the x86 box, fixed slices Verified-30 + Multilingual-28 (58 instances), -w 4,
reasoning_effort medium, 1.0/0.95/20, max_tokens 16000, step limit 250 (benchmark default).
Smoke (base, 2 instances): 183 s gen, 1 resolved, 1 context overflow (config, fixed).
H: resolve rates of the three arms within the run-to-run spread of prod (expected spread +-3 instances of 58 at
temperature 1.0); a cut is a NOTICEABLE loss only if both of its starts sit below both prod starts AND paired
per-instance losses outnumber wins clearly (sign test p<0.1 over the 116 paired instance-runs).
bf16ssm is the arm where long trajectories can show state-error accumulation; watch its empty-patch and
ContextWindow/format-error counts separately from resolve rate.
Out of range: any arm with > 2x prod's empty patches or error exits -> triage the trajectories before any verdict.

## TensorFold + EXL3 3.05 bpw on the 58 slice (2026-09-30, user: "do the 58 swt slice on tensorflow/exl3 (start slow and watch)")

Server: TensorFold 0.5.0 (local branch `pr-fp8block`; the EXL3 path is stock), `turboderp/Qwen3.8-Flash-Next-exl3` @
`69e33439` (3.05 bpw, 5-bit n-gram table, 25/25 files verified), `tensorfold serve … --name flashnext --context 65536
--max-tokens 16000 --parallel 4` (eager: no decode graphs under `--parallel`; MTP 1–6 drafts, 30 % confidence stop).
Client unchanged from §5u: `/root/fn-swe/run.sh` (mini-swe-agent 2.4.5 native `bash` tool calls, `fn.yaml` sampling
1.0/0.95/20, reasoning_effort medium, max_tokens 16000, 4 workers), same Verified-30 + Multilingual-28 slices.
Reference: vLLM prod on our FP8/NVFP4 checkpoint, 48 and 52 of 58 (§5u).
- H (quality): EXL3 at 3.05 bpw is a larger precision cut than any §5u arm (4.5-bit experts → ~3 bits overall), so
  resolved 40–50 of 58; a result inside 48–52 means no noticeable loss at this dose.
- H (tool calls): TensorFold's tool-call parsing works for mini-swe-agent (checked: two parallel `bash` calls, valid
  JSON); format errors per trajectory no higher than on vLLM. The smoke (2 instances) gates the full slice.
- Not comparable on speed: eager decode under `--parallel`, different engine.
