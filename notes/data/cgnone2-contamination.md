# cgnone2 — contamination note (2026-09-08, model-authored)

Arm 1 (`piece1`, started 07:37) ran its c=4 and c=16 reps between roughly 07:45 and 07:54.
During exactly that window I ran heavy **host** work on the box: repeated python parses of two
multi-GB session transcripts (45 MB + 33 MB, read whole into memory), file copies, git operations.
Two of my own background shells were killed by the low-memory guard at ~07:51; `free` showed
1 GB free with `some avg10=31`.

Symptoms visible in the arm-1 data:

- `piece1 DV c=4` spans **64.1 / 37.5 / 37.1 tok/s** — a 1.7x spread within ONE server start.
  Reps inside a single start should not do that; restarts are where our spread lives (det-162).
- `piece1 DV c=16 rep=1` reports `garbage 1/16`.

**Handling.** Treat `piece1`'s c=4 and c=16 cells as contaminated and take those cells from the
later PIECEWISE arms (`piece2`, `piece3`). `piece1 c=1` ran 07:37–07:45, before the host work, and
is probably clean but is the weakest of the three starts. The c=16 garbage stream is **not**
evidence of a cudagraph-mode effect until it reappears in a clean arm.

**Rule this reinforces.** No heavy host IO or memory work while a benchmark arm is running — the
GPU being idle-looking says nothing, because this box shares one memory pool. Same mistake class as
the `du`/`find` sweep that landed inside `cgsize2` start 3.
