DRAFT — needs the user's go. ashhart/TensorFold #180 reply to plotarmordev's two-Spark review (2026-10-01), together with
a push of their test commit (cherry-picked, their authorship).

Thanks for the run and the test fix. I hadn't run `test_flashnext_tp_multi.py`, and you're right that it still
asserted the old swap. Your commit is on #180 now as yours ([`d2e651a`](https://github.com/jschmied/TensorFold/commit/<FULL>)).
On one GB10 the set you listed passes: <RESULT>.

The single-request gap left on two ranks is the swap: rank 0's plan names kept prompt ends by slot, so
`_relocate_kept` only runs on one GPU. Carrying "move this kept end to that slot" in the plan, for rank 1 to replay
like a resize, would close it. That's a separate change, if wanted.

Written with AI assistance (Claude Code).
