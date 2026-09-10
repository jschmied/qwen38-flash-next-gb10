DRAFT — user go (same). New issue on pangoleen/qwen3.8-flash-next-dgx-spark (2026-09-10).

TITLE: `01-draft-vocab`: 65,536 is probably larger than the vocabulary that actually occurs — we measured the optimum at 16,384

Measured on a different serving route, so treat this as "worth testing there", not "your number is wrong".
Our stack: single GB10, vLLM nightly, `RadixArk/Qwen3.8-Flash-Next-NVFP4` with an FP8 head, MTP 3, PLE via
the CPU-offload worker (vllm#53899) rather than the NVMe mmap. Same model, same chip, different plumbing.

**The observation.** We built the draft-vocabulary ranking from our own agent traffic and then measured how
large the *observed* vocabulary is: **48,476 distinct token ids in total**, with held-out output using only
**2,048 distinct ids**. A 65,536-token slice is therefore larger than the vocabulary that occurs at all —
it cannot be restricting much, and your own numbers say so: acceptance **3.64 → 3.57 accepted per pass**,
i.e. essentially unchanged. That is what a non-binding slice looks like.

**The sweep.** `full` / 4,096 / 8,192 / 16,384 / 32,768, three server starts each, c=1 (8 reps) and c=4
(3 reps), compared rep-against-rep with ranges rather than means:

| comparison, matched reps, warm only | wins | loses | overlap | gain where it wins |
| --- | --- | --- | --- | --- |
| **16,384 vs full** | **5/7** | 0 | 2 | **+8.5 … +12.1 %** |
| 16,384 vs 32,768 | 4/7 | 1 | 2 | +1.6 … +12.0 % |
| 32,768 vs full | 5/7 | 0 | 2 | +5.0 … +6.8 % |

Acceptance tracks the slice monotonically and the coverage limit starts biting at **8,192** (3–13 pp lost)
and is severe at **4,096** (8–23 pp) — 4,096 was both the least accurate *and* the slowest arm, so the
cheaper head never paid for the lost acceptance. 16,384 tracks `full` to within about a point at almost
every rep.

**So the lever is real and you have it, but 65,536 is probably leaving most of it on the table** — our
32,768 arm is worth +5…+6.8 % over unrestricted, and 16,384 a further step beyond that. One caveat that may
or may not transfer: our ranking comes from *our* agent traffic, and the right slice depends on the
vocabulary your workload actually touches. The cheap check on your side is the same one we ran — count the
distinct ids your corpus produces before choosing the cut.

Two smaller notes while reading the repo. Your framing that "the output the server produces cannot change;
only the guessing gets cheaper" is exactly right and worth keeping prominent — the drafter proposes and the
target verifies, so this is a pure speed knob with no quality axis. And your `03-staged-ple` argument
transfers to our stack in a way you might find useful: on the nightly `qwen4_exp` module, the engine's own
`splitting_ops` list contains `vllm::qwen4_exp_compute_ple_ngram_ids` and `vllm::qwen4_exp_ple_short_conv`,
i.e. the PLE lookup and short-conv are *declared* graph-partition points. That is your "a mid-forward read
cannot be recorded" stated by the build itself, on a different PLE implementation.

Happy to share the vocabulary-coverage script or the sweep harness if either is useful.

*AI assistance was used in preparing this issue; the measurements are ours and were reviewed before posting.*
