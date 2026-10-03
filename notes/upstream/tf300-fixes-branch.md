POSTED 2026-10-03 (user: "post to 300"). TensorFold PR #300 comment (2026-10-03).

@jayleaton I reviewed #300 and fixed what I found on a branch on top of `5cbe389`, in case it's useful to you: [compare 5cbe389...pr300-fixes](https://github.com/jschmied/TensorFold/compare/5cbe389...pr300-fixes). It has 18 commits, each with tests that failed before the fix. Take or cherry-pick whatever fits.

**Fixes**
- **Ranks agree on each step.** Ranks now vote on ADMIT, EVICT and ROUND. An admission that fails on any rank is undone on every rank, cache order included. Rank 0 also forgets a lower-tier state another rank couldn't resume from. ([36b531f](https://github.com/jschmied/TensorFold/commit/36b531f), [7c35010](https://github.com/jschmied/TensorFold/commit/7c35010), [3670830](https://github.com/jschmied/TensorFold/commit/3670830))
- **Tiered cache.** Entries pushed out of the host tier move down to disk instead of being dropped. The entry a lane resumes from is pinned by identity, not by id list. Lookups hash all prefix lengths in one pass. ([8ab84ea](https://github.com/jschmied/TensorFold/commit/8ab84ea), [fc11c5b](https://github.com/jschmied/TensorFold/commit/fc11c5b))
- **Spills.** A spilled state carries only its complete rows. Pages are copied without plane-sized temporaries. A failed write leaves no `.tmp` file. Truncated or oversized headers and files that can't be deleted stay inside the disk tier, and a failing disk never stops serving. ([f42791a](https://github.com/jschmied/TensorFold/commit/f42791a), [d1fc34b](https://github.com/jschmied/TensorFold/commit/d1fc34b), [6153235](https://github.com/jschmied/TensorFold/commit/6153235))
- **Candidates.** Each verify window asks for the candidate count its own sampling needs, and sampling reads only those columns. ([89923c6](https://github.com/jschmied/TensorFold/commit/89923c6), [781656b](https://github.com/jschmied/TensorFold/commit/781656b))
- **Admission. This one changes behaviour you chose:** with nothing live, a request that can't fit is no longer started below the memory floor. Host-kept states are dropped first, then the request is refused with a ValueError. Two tests were changed to match. ([57047c4](https://github.com/jschmied/TensorFold/commit/57047c4))

**Optional additions.** None changes a caller that doesn't use it.
- `LaneForward.mixed()` runs a round's prompt pieces and its decoding lanes' windows in one forward, so a MoE reads its experts once. ([e758ee5](https://github.com/jschmied/TensorFold/commit/e758ee5))
- `Stream.stops` (message starts) keep states on the grid below them, so another conversation resumes from a shared system prompt. ([c4b321a](https://github.com/jschmied/TensorFold/commit/c4b321a))
- `drafter.attach(forward)` gives a drafter that reads model state, such as an MTP head, its rank's forward. ([f9407e2](https://github.com/jschmied/TensorFold/commit/f9407e2))
- `decode_share` sizes prompt pieces next to decoding lanes from timed rounds (prompt time fitted as a fixed plus a per-row cost; clock injectable). ([db66c35](https://github.com/jschmied/TensorFold/commit/db66c35))
- `Stream.cancelled()` ends a stream before its next round, so a prompt stops filling once the client leaves. `Scheduler.submit` takes it; the server isn't wired to it yet. ([43303a3](https://github.com/jschmied/TensorFold/commit/43303a3))

**Tests.** Your suites plus the new tests: 222 pass on the CPU. `tests/cuda/test_cuda_rowgraphs.py` and a new `test_cuda_kvpool_copies.py` pass on one GB10. The wider host suite fails the same 45 tests on this branch as on `5cbe389`, so none are new; many of them need MLX.

**Not done:** asynchronous disk writes, demoting entries to a lower tier when trimming instead of dropping them, an all_reduce in the communicator, and a two-rank measurement of the vote's cost.
