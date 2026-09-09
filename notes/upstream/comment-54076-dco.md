DRAFT — user go "post". vllm-project/vllm PR #54076, comment (2026-09-09).

@wickist — mechanical rather than substantive, and it may be why this has sat: **DCO has been failing
since the PR opened**, so CI never gets past it.

None of the three commits carries a `Signed-off-by:` line:

```
9548d605  [Test] Reproduce mamba align split with heterogeneous KV block sizes
10e28b6c  [Bugfix][V1] Use the mamba cache group's block size for align-mode chunk splitting
244edeed  [Bugfix][V1] Stop mamba align chunks at every crossed state boundary
```

`DCO` shows `action_required` and `pre-run-check` fails, which skips `Check format` and `pre-commit`
behind them. `git rebase --signoff main && git push --force` should clear it.

Worth flagging because DCO surfaces as a check rather than as a review comment, and it is easy to miss
while resolving merge conflicts — which you have now done twice. Fourteen comments here and no human
review submitted yet; a red DCO is a plausible reason reviewers have not picked it up.

*AI assistance was used in preparing this comment.*
