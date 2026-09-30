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

### Same-day vLLM baseline (`vllmnow1`, queued after TensorFold run 1)

User: "but our old swt run was before many speed optimizations". Today's prod config (drop-ins 20–60: K=5 +
probabilistic drafting, NVFP4 draft head, RecoverSSM + F4, pmu 64, bf16 SSM, HC fusion, Triton prefill MoE + GDNNQ) on
our FP8/NVFP4 checkpoint, test unit on :8092, 64k context, 8 GiB KV (4 agents × 64k), same client and slices. H: resolved
48–52 of 58 (§5u's range; the §5ag/§5aa changes do not change text quality beyond drift); gen time per slice below §5u's
(2,345–2,999 s Python, 2,906–3,929 s Java/JS) by 10–20 % from the prefill work. Compared with TensorFold+EXL3 run 1 on
wall-clock per slice and resolved count; one run each, so only large gaps count.

### Queued after the baseline: TensorFold + EXL3 run 2 and the GSM8K/HumanEval screen (2026-09-30, user: "yes")

User read §5aj as "EXL3 is better quant than NVFP4"; one run cannot separate 52 from vLLM's 49–53 (overflows counted as
solved). Run 2: same server flags, same client. H: 49–54; the pair's mean within ±2 of vLLM's six-run mean (51.9 with
overflows counted). Logprob divergence against BF16 is not possible on TensorFold's CUDA Flash-Next engine (no
logprobs, TensorFold#108 open), so the screen is `evalprobe.py` (GSM8K 1319 + HumanEval 164, greedy, thinking off,
c=16 against `--parallel 4`), compared by McNemar with evalgq-base0/1 (vLLM prod, 95.91 / 96.21 % GSM8K, 95.12 / 95.12 %
HumanEval). H: within those starts' spread; a drop of more than 1.5 pp GSM8K with McNemar p < 0.05 against both base
starts is a real loss at 3.05 bpw.

## Hard slice (2026-09-30, written before the run)

User: "replace 10 easy with 5 harder and 5 hardest" (x86 disk space). Verified has only three ">4 hours" instances, so
the 10 are those three plus seven "1-4 hours" (random, seed 1): `set-hard.txt`. Dropped: ten "<15 min fix" instances
every run so far solved (tfexl3r1, vllmnow1, tfexl3r2 Python), `set-python-dropped.txt`; the mixed Python slice is the
20 kept + these 10 (`set-python-mixed.txt`), scored from the existing runs plus the hard runs. Two runs per engine,
same servers, client and sampling as §5aj.

- Hard 10, per run: **2–6 resolved** (20–60 %); fewer than 2 on both engines means the slice is too hard to
  separate them, more than 7 means the labels do not track difficulty for this model.
- Engines: within ±2 per run, i.e. no separable gap at n=10 per run; a gap has to hold in both runs to count.
- Gen time per run: 25–60 min at 4 workers; context overflows 0–3 per run (longer trajectories than the easy slice).

## Java/JS swap and the combined plan (2026-09-30, before the run)

User: "yes, prune after current run and swap slices". Multilingual has no difficulty labels; proxy = reference-patch
size. Our 28 Java/JS were all one file, ≤ 11 lines (the pool in the same 9 repos: median 8, max 219 lines, up to 8
files). Dropped: the 10 smallest-patch instances both tfexl3r1 and vllmnow1 solved (`set-javajs-dropped.txt`); added:
the 10 largest reference patches in the same repos (`set-jshard.txt`, 23–219 lines). Chain `mixswe.sh.txt` replaces
the queued vllmswe3 + hardswe (stopped before they started): vLLM run 1 = vllmnow1 kept + vllmhard1; vLLM run 2 =
vllmmix2 (full mixed 58); TF runs = tfexl3r1/2 kept + tfhard1/2.

- Java/JS hard 10, per run: **3–7 resolved**; patch size is a weaker difficulty proxy than Verified's labels, so
  expect less of a drop than on the Python hard 10.
- Mixed 58 per run: **40–50** (from 51–52), engines within ±3; a gap counts only if it holds in both runs.

## Overflow reruns at 131072 context (2026-09-30, before the run)

User: "re run failed with bigger context". Every trajectory of vllmhard1, vllmmix2, tfhard1, tfhard2 that ended in
`ContextWindowExceededError` at 65,536 is re-run at 131,072 (vLLM KV 12 GiB; TensorFold 4 slots, 2 if 4 do not fit),
same engine, sampling, per-step max_tokens 16000 and seed config; the rerun's result replaces the overflow in §5al.
So far 6 of 20 vLLM hard instances overflowed (2 Python, 4 Java/JS).
- Resolved among the reruns: **25–50 %**, well below the 100 % that "count an overflow as solved" assumes; a trajectory
  that needs > 64k is usually one that is stuck, not one that is nearly done.
- A few (0–2 of 6) overflow again at 128k.
