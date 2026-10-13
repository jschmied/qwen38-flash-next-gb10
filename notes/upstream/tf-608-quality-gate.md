# #608: mechanical quality gate (BobClawblaw's ask) - result + reply draft (NOT posted)

2026-10-13. BobClawblaw on #608 (08:11): prefers C; make the quality check mechanical with fixed limits (NLL within
0.005 of Triton, argmax >= 99 %, KL <= 0.002), offers an RTX 5090 (sm_120) rerun.

## Built (fork tf-q27, uncommitted; same runNll in tf-608 for the reference dump)
`tf-qwen27-cuda nll MODEL --prompts corpus.json [--dump F] [--against F]`: corpus = {"name": [ids]}; per doc + total
NLL/top-1, against a reference dump: KL(ref||ours) mean/max, argmax agreement, largest flip (reference logit gap
between its argmax and ours), gated, exit 1 on FAIL. Diff: data/q27qual/nll-gate.diff. Corpus ~/q27/qual/corpus.json
(4 x 1,024 tokens: docs, code, German, chat) - LOCAL ONLY (German doc is from private notes; replace before sharing).

## Result (data/q27qual/*.log), 4,092 rows vs Triton kset3
| build | dNLL | argmax agree | KL mean | largest flip | Bob's limits | kernel test |
|---|---|---|---|---|---|---|
| native | +0.0011 | 99.12 % | 0.00071 | 0.25 | pass | pass |
| M1: RoPE freq x(1+1/256) | +0.0012 | 99.07 % | 0.00103 | 0.375 | **pass** | FAIL (worst 58,563x tol) |
| M2: q/k norm divisor D-1 | +0.0008 | 99.17 % | 0.00070 | 0.25 | **pass** | FAIL (worst 1.43x) |

Reference reproduces the earlier docs number exactly (0.8029). Per doc, German argmax is 98.53 % on the clean build:
a 99 % bar must apply to the total, not per doc.

**Finding:** the teacher-forced gate cannot tell the clean kernels from two real bugs - ulp-level rounding already
moves argmax ~0.9 % and KL ~7e-4, and these bugs sit inside that floor. The float64 kernel tests catch both. So under
C the layers are: kernel float64 bounds + mutations (catch bugs), engine invariants (drafted == serial, concurrent ==
alone, resumed == fresh), and the model gate as an integration check (catches gross breakage: wrong weights, a
skipped layer, wrong layout), not as the thing that keeps "close" from drifting.

## Reply draft (one topic, short)

@BobClawblaw agreed, and I built it to see how sharp it is. `tf-qwen27-cuda nll --against <reference dump>`:
teacher-forced over 4 documents (docs, code, German prose, chat; 4,092 rows), reporting NLL, argmax agreement,
KL(Triton || native) and the largest flip, with fixed limits and a non-zero exit.

Native vs the Triton set: NLL +0.0011, argmax 99.12 %, KL 0.00071, largest flip 0.25 logits (2 bf16 ulp). Within your
limits on the total (one document alone is 98.53 %, so the 99 % has to be the total).

Then I checked it catches something. Two deliberate bugs in `qk_rows`: RoPE frequencies off by 1/256, and the q/k
norm dividing by D-1. Both pass the same gate (+0.0012 / 99.07 % / 0.00103, and +0.0008 / 99.17 % / 0.00070). Both
fail `tf-cuda-test qk-rows` at once (worst 58,563x and 1.43x the float64 tolerance).

So at this length the model-level numbers sit on a floor of ulp noise, and real bugs hide in it. I'd keep it in the
gate as an integration check (wrong weights, a skipped layer, a layout mix-up move it a lot), but what stops "close"
drifting is the per-kernel float64 bound with a mutation per kernel, plus drafted == serial and concurrent == alone.
A second chip would be useful for exactly that: the 5090 run checks the kernel bounds on sm_120, where tiles and
`ex2` differ.
