DRAFT — user go "ok do 1". vllm-project/vllm PR #54076 and PR #53798, comment (2026-09-09).

Closing out the correction I promised here on 2026-09-03, and the answer is that I am **withdrawing the
number rather than restating it**.

**What I posted on 09-02** — "this patch alone takes healthy-acceptance turns from 44 % to 15/16" — used
a healthy/broken per-turn metric that I later found measures the harness, not the patch. Our agent loop
sent `ignore_eos: true, max_tokens: 130` while the model's real answer was 30–40 tokens, so every turn
continued past its end-of-turn token into one of two near-tie continuations: a chat-template restart,
which the drafter predicts at ~100 %, or a wall of `<|im_start|>`, which it never predicts. "Healthy" and
"broken" turns were those two filler modes. The metric is a property of `ignore_eos`, and it does not
measure the align defect.

**Why I am not supplying a replacement effect size.** I re-ran the grid EOS-correctly and have a clean
*unpatched* baseline — acceptance by MTP n = 60 / 58 / 46 / 40 / 37 / 29 / 26 % for n = 2..8, three
starts agreeing, and I have confirmed that arm was genuinely unpatched from the runner's own preflight
gate (it aborts if `mamba_state_block_size` is present in the scheduler, and it logged OK). But I have
**no EOS-correct patched arm** to pair it with. So there is no corrected before/after, and "the direction
stands" from my 09-03 note is not something that data supports either. Both figures are withdrawn.

**What is unaffected**, and is why I still think this PR is right: the defect is a code fact, not a
benchmark result. The align-mode split used the QSA ring capacity as its unit instead of the mamba
state block, so on this configuration every prefix-cache resume continued from a GDN state up to one
mamba block stale — silently, with no error. In an agent loop every turn is a resume. That argument
stands on the code and does not depend on any number I posted.

**Offer.** This PR has been blocked on conflicts twice in three days with no reviewer, and I would rather
hand a reviewer a real measurement than an anecdote. We have a GB10 (sm_121, TP=1) and the EOS-correct
harness. If you name the cell you want — acceptance, TTFT, or prefix-cache hit rate, patched vs
unpatched, however many starts you consider enough — we will run it and post it with the raw data,
whichever way it comes out.

Apologies for the six-day gap on a correction I said would follow immediately.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
