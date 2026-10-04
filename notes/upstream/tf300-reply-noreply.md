POSTED 2026-10-04 (user: "ok, noreply address"). TensorFold PR #300 reply to jayleaton (2026-10-04).

@jayleaton thanks for taking them, and for the careful read.

Dropping the vote from ROUND is right. A follower that fails inside a round's collectives would leave rank 0 waiting where the vote's all-gather can't match, and the vote's per-step cost isn't worth it. Putting the status in the follower's slot of the next head gather sounds good as a separate PR.

Please re-author my commits to `600316+jschmied@users.noreply.github.com` before this lands. I'll use that address from now on.

I'll open the follow-ups as separate PRs on top of this one, starting with `Stream.cancelled()` and `Stream.stops`, which don't change speed.
