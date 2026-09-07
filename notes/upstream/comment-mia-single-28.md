DRAFT — user go given ("yes, comment there"). MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#28 (2026-09-07).

Independent confirmation, and we have root causes for it. Same box class (DGX Spark, GB10, sm_121, TP=1), same
symptom, and on the image you name — `0.1.dev20073+g8e685d198` — it is **three separate kernel defects**, not one:

1. **The NVFP4 MoE finalize reduces in a non-deterministic order.** Fixed by vllm-project/vllm#54948
   (with #54945); before that, identical inputs give different sums for the same expert combination.
2. **The GDN align-block seeding** — vllm-project/vllm#54076 and #53798.
3. **`persistent_topk`** hands out output slots with `atomicAdd`, i.e. in thread-arrival order, and takes
   exact-key ties at the last radix round first-come. The QSA sparse attention sums the selected keys **in
   output order**, so the hidden state forks. Still open as vllm-project/vllm#55122.

**Why "also with MTP off" is expected rather than puzzling:** two of the three sit outside the speculative
path entirely, so disabling MTP cannot remove them. And your 27B being byte-identical on the same box fits —
it does not take the QSA indexer or that MoE path.

With all three fixed, strictly sequential greedy on an idle server reproduces **bit-for-bit, 6 of 6 across
server restarts**, on this hardware. Concurrent greedy is a separate question and we do not claim it here.

One methodological note that cost us several days. **A "same prompt twice" determinism check can be blind to
a whole class of these bugs.** We had a defect where every forward consumed the *previous* step's data; two
identical consecutive requests agreed perfectly, and only a **cold request compared against a warm one**, or a
request that followed a *different* request, exposed it. That specific defect is in a PLE-offload path your
image does not carry, but the test shape is worth adopting regardless.

**On the acceptance figure, which we think is the more consequential half of your report.** You note
acceptance ≈0.93 on long prose and attribute it to recitation. Your own sweep table corroborates that: at
S=1 it reads `tokens/step` **3.00** with MTP 3 — every step accepting all three drafts — while S=2/4/8 read
2.81–2.84 and the per-position acceptance quoted just below (0.80 / 0.59 / 0.41) implies ≈2.80. We treat
**accept-length pinned at the maximum as a corruption or recital signature rather than health** — we once had
a field case reading 3.00/3 while the same build scored 0/10 on GSM8K. Worth separating "the drafter is
accurate" from "the model is repeating its context" before any single-stream prose throughput number is
quoted, ours included.

Happy to share the standalone test we use for the top-k kernel (bit-identity across repeats plus equality to
an exact reference under tie-heavy inputs) if it is useful for checking a build.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
