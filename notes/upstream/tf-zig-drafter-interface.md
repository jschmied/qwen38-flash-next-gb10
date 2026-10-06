POSTED 2026-10-06 as ashhart/TensorFold#455 (draft PR against zig-flashnext). Title: lanes: an external drafter behind any target backend

Today a drafter lives inside its family's backend: `Backend.draft` is that family's MTP head. Drafters that read the
target's hidden states from outside (EAGLE-3, a DFlash block, an MTP head loaded on its own) have no place to go. Two
cases I know of: Kolibri-1, which has no MTP head and which we serve on our fork with an EAGLE-3 drafter
(`josch15366/Kolibri-1-EAGLE3-drafter`, about +25 % single-stream decode, byte-identical), and Laguna-S-2.1 with
poolside's published DFlash drafter (#312).

This PR adds the interface in the lane core only. No family changes; Nemotron's and Flash Next's own heads are untouched.

## What changes

- `backend.zig`: `Features` and an optional `Backend.features(s, taps, start, count)`, the tapped states of cache rows
  the target still holds: the prompt rows its pass computed (from `s.cached`) until the first verify, then the last
  verify's rows, and after a keep the rows it retained (a single-stream round keeps before it drafts), until the next
  verify, prefill or release. Packed rows in the drafter's `taps` order, `space` (host, or a device pointer on `device`,
  which the drafter must run on), `dtype`, and `ready`, a tagged fence (`none`, `cuda_event`, `metal_event`). Null by
  default (`error.NoFeatures`).
- `drafter.zig` (new): `Drafter` vtable `taps`, `facts`, `open`, `absorb`, `hold`, `held`, `release`. Every call takes a
  round's streams together (`[]Absorb`, `[]Hold`, one `held` readback), so a GPU drafter runs them as one batch.
  `Facts` carries the drafter's own depth, step cost, prior, plain guard and batching.
- `drafted.zig` (new): `Drafted`, a target backend plus a drafter as one `Backend`, so the round loop is unchanged.
  - A stream with drafts off never reaches the drafter: `"draft": false` on the wrapped backend is the target alone.
  - Prefill absorbs the prompt rows the pass computed; each round absorbs the kept rows, then holds drafts; held drafts
    reach the target as host tokens.
  - The drafter is only asked to draft once it has the row before its pending token. A prompt restored whole has none:
    that one round takes fillers (the pending token repeated, which the target rejects), and the drafter starts after
    the first verify.
  - Lifecycle: the wrapper tracks the streams it opened (the slot is reserved before `open`); a failure after the
    target's prompt pass releases target and drafter there (the core releases only a cancelled pass, and then the
    drafter was never opened).
  - A steady round allocates nothing: the per-round lists are kept.
  - `facts()` takes the depth rule's step cost, prior, plain guard and batching from the drafter, not the target.
  - Chains only: trees and early speculation stay with heads built into a family.
- `fake.zig`: the fake target exposes `features` (a row's state is its token).

## Tests (`drafted_test.zig`, host only)

All drafted runs go through a strict test target: the fake with its own state snapshot, holding exactly what `Features`
promises (a keep overwrites the rows it drops) and refusing anything else. The fake drafter refuses to draft without
the row before its pending token.

- Drafted rounds equal one-token rounds: greedy, sampled, and two streams sharing rounds (and those reach the drafter as
  one hold of both streams, at most one readback a verify).
- The fake drafter's drafts land far above chance (it drafts the target's token four times in five; a row or position
  mistake would drop it to about 1 in 97). Shifting the absorbed rows by one fails.
- Drafts off on the wrapped backend: the same tokens as the target alone, and no drafter call at all.
- A prompt restored in part and whole, greedy and sampled: the one-token decode, the drafter starting past the restored
  rows.
- Injected failures (drafter open, absorb, a target without features, a cancelled prompt pass): target and drafter are
  each released once, the drafter only if it was opened.
- The drafter's facts reach the depth rule in place of the target's.

Removing the drafts-off bypass, the cleanup, the whole-restore check or the fillers each fails its test; reading the
verify's rows shifted by one fails three tests on the strict target (`RowsNotHeld`). `zig build test` passes on
zig-flashnext 7742ddd (GB10, Zig 0.17.0).

## Not in this PR

- No CUDA target exposes `features` yet. The first would be a Kolibri-1 adapter on the CUDA registry (#443) with our
  drafter behind it; a registry hook so an entry can load a drafter belongs with it.
- The depth rule prices drafts linearly (`step_ms x drafts`); a block drafter (DFlash) costs one pass for any depth, so
  a cost-by-depth table in `depth.zig` should come with the first measured block drafter.
- The drafter's own state is not part of prompt reuse: after a restored prefix it drafts without those rows, and after
  a whole restore it starts one round late. Keeping the last tapped row with a kept prompt state would close that.

## Questions

1. Is `Backend.features` the right boundary, or would you rather the target hand states to the drafter inside `verify`
   (one call fewer per round, but `verify` grows a second job)?
2. Held drafts go to the target as host tokens (one device-to-host copy a round). A device-side path would need
   `Window.held` to name the drafter's buffer; worth it now, or after a CUDA target measures the copy?
