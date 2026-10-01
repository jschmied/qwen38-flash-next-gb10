POSTED 2026-10-01 as https://github.com/ashhart/TensorFold/pull/175 — user's go 2026-10-01 ("post whats ready"). ashhart/TensorFold PR from jschmied/TensorFold:pr-seed-salt.

TITLE: TENSORFOLD_SEED_SALT: independent repeats of an evaluation without changing the client

A request without `seed` samples with a seed [drawn from its prompt](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/cuda/server.py#L290)
(`exact_sampling.seed_for`), so the same conversation always samples the same reply. That is the point of it, but it
also means an evaluation run twice against one server mostly repeats itself. We ran SWE-bench through
mini-swe-agent (which sends no seed) twice on the same TensorFold server: 48 of 58 patches came out byte-identical and
the same 52 instances were solved; a trajectory matched message for message until a tool output differed. The second
run measured nothing new, and the run-to-run spread came out as zero by construction.

`TENSORFOLD_SEED_SALT` (an integer; unset or 0 keeps today's seeds exactly) is mixed into every prompt-derived seed
through `seed_for`'s existing `salt` argument. A different salt per run gives independent repeats; the same salt
reproduces a run. A malformed value is refused at startup. Requests that send `seed` are unaffected.

- `src/tensorfold/engine/exact_sampling.py`: `SEED_SALT` read once from the environment; `seed_for(tokens, salt=None)`
  uses it when no salt is passed.
- `docs/api.md`: the `seed` row and a paragraph under the exactness notes on repeated evaluation runs.
- `tests/test_seed_salt.py`: the default keeps today's seed, a salt moves it and stays reproducible, parsing, refusal.

CPU suite: the same failures as main on this machine (missing MLX and similar), none new.

Written with AI assistance (Claude Code); the author reviewed every change.
