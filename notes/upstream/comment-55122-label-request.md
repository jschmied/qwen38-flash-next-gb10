DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122 (2026-09-07). Posted by the human author, not the agent.

@LucasWilkinson — you reviewed and merged #54110, the last change to this kernel, so you have the
context. Could you (or another maintainer) add the `ready` label? `pre-run-check` gates `pre-commit`
on it, and the alternative it accepts — four merged PRs — I don't have yet (one so far, #55180).

The PR is finished from my side: `pytest -k persistent_topk` is 221 passed / 26 skipped, the
standalone determinism grid is 210 / 210 against 0 / 210 for the current kernel, and a server-level
A/B with three starts per arm shows no end-to-end cost. The body states the hardware risks and the
one ~5 % regression I could not explain, rather than leaving them for a reviewer to find.

One thing worth fixing independently of this PR: `csrc/libtorch_stable/` has no CODEOWNERS entry
(also noted on #51782). The six reviewers here were pulled in by the `/tests/kernels` line, not by
the kernel directory the change is actually in.
