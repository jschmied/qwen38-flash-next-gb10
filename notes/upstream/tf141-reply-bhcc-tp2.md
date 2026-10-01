POSTED 2026-10-01 (user: "yes, both"). ashhart/TensorFold #141 reply to BHCC2025's two-rank run (2026-10-01), together with a
push of the follow-up commit to #180.

Thanks for the two-rank run. It found a bug in #180. On two ranks, rank 0 plans on `Shadow`s of the slots, and
`dec.solo.st` is itself a `Shadow` there. `_is_solo` unwrapped only its argument, so it was never true while planning.
#180 therefore changed nothing on your pair except the warm-up order. That also means the `cb5101d` and `+ #180`
columns ran the same path, so their −5…+2 % spread is run-to-run variation, not an effect.

The lone-request gap on `--parallel 8` (≈11 %) looks like the same recapture we measured on one GPU (+35–50 % there,
with 512-token replies). It's smaller on yours, presumably because the replies are longer per capture.

Follow-up on #180: `6ada832`. `_is_solo` now compares the slots the Shadows stand for, and an idle graph slot can be
released under memory pressure while planning too (recorded for rank 1 like any other resize). It adds a host test
through a `Shadow`, which fails on `6a3a8b1`. One GPU only here. Would you rerun the 1-user cells (code and chat,
greedy) on the pair with it?

Written with AI assistance (Claude Code).
