POSTED 2026-10-06 as ashhart/TensorFold#455 (draft PR against zig-flashnext). Title: lanes: an external drafter behind any target backend

Today a drafter lives inside its family's backend: `Backend.draft` is that family's MTP head. Drafters that read the
target's hidden states from outside (EAGLE-3, a DFlash block, an MTP head loaded on its own) have no place to go. Two
cases I know of: Kolibri-1, which has no MTP head and which we serve on our fork with an EAGLE-3 drafter
(`josch15366/Kolibri-1-EAGLE3-drafter`, about +25 % single-stream decode, byte-identical), and Laguna-S-2.1 with
poolside's published DFlash drafter (#312).

This PR adds the interface in the lane core only. No family changes; Nemotron's and Flash Next's own heads are untouched.

## What changes

- `backend.zig`: `Features` and an optional `Backend.features(s, taps, start, count)`, the tapped states of cache rows
  the target still holds: the prompt rows its pass computed (from `s.cached`, so a prompt restored by prompt reuse is
  fine) and the last verify's rows until the next verify. Null by default; `features()` returns `error.NoFeatures`.
- `drafter.zig` (new): `Drafter` vtable: `taps`, `depth`, `open`, `absorb(features, start, follow)` (rows from `start`
  replace what the drafter held there, so a rollback is the next absorb's `start`), `hold(position, depth)`, `held`,
  `release`.
- `drafted.zig` (new): `Drafted`, a target backend plus a drafter as one `Backend`, so the round loop is unchanged.
  Prefill absorbs the prompt's computed rows; each round absorbs the kept rows, then holds drafts; held drafts reach the
  target as host tokens. Chains only: trees and early speculation stay with heads built into a family.
  `facts()` derives the round loop's `Model` from the target's.
- `fake.zig`: the fake target exposes `features` (a row's state is its token).

## Tests (`drafted_test.zig`, host only, on the fake target)

- Drafted rounds equal one-token rounds: greedy, sampled, and two streams sharing rounds.
- The fake drafter's drafts land far above chance (it drafts the target's token four times in five; a row or position
  mistake would drop it to about 1 in 97). Shifting the absorbed rows by one fails both tests.
- A prompt restored from a kept state: the drafter's first row is the restored one, and the output equals the
  one-token decode, greedy and sampled.
- A target without `features` refuses (`error.NoFeatures`).

`zig build test` passes on zig-flashnext 7742ddd (GB10, Zig 0.17.0).

## Not in this PR

- No CUDA target exposes `features` yet. The first would be a Kolibri-1 adapter on the CUDA registry (#443) with our
  drafter behind it; a registry hook so an entry can load a drafter belongs with it.
- The drafter's own state is not part of prompt reuse: after a restored prefix it drafts without those rows (the target
  still verifies every draft, so only speed is affected).

## Questions

1. Is `Backend.features` the right boundary, or would you rather the target hand states to the drafter inside `verify`
   (one call fewer per round, but `verify` grows a second job)?
2. Held drafts go to the target as host tokens (one device-to-host copy a round). A device-side path would need
   `Window.held` to name the drafter's buffer; worth it now, or after a CUDA target measures the copy?
