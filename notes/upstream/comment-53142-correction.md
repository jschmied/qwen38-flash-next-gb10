# Correction to post on vllm#53142 (and the one-liners on #54076 / #53798) — POST AFTER MTPGRID3

The healthy-turn percentages I posted (44 % unpatched → 88 % patched) came from an agent loop that
sent `ignore_eos: true` with `max_tokens: 130`; the model answers in 30–40 tokens, so most of every
measured turn was post-EOS filler, and "healthy vs broken" was which filler the target picked
(finding 59). The *direction* stands (the wrong block-size units change the resume state and thus
the drafter's inputs), the numbers do not. Replace with the EOS-correct grid (`MTPGRID3`):

- unpatched vs patched acceptance on REAL tokens: <fill from MTPGRID3 + a matching unpatched run>
- note the residual is 0: the only remaining nondeterminism was `persistent_topk` (#54521), fixed
  separately (kernel diff in patches/kernel-det, exact-selection workaround in qsafix_patch.py)

Keep it to five lines. Say explicitly that the earlier numbers were a benchmark artifact.
