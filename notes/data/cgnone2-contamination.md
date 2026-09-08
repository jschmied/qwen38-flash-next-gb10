# cgnone2 — "contamination" note, RETRACTED (2026-09-08)

**The original claim was wrong and is retracted.** I wrote (08:0x) that arm 1 (`piece1`) was
contaminated by heavy host work I ran during it — repeated python parses of two multi-GB session
transcripts, which did push the box into the low-memory guard and killed two of my own shells. The
inference from that to "the arm is spoiled" was unsupported, and arm 2 refutes it.

## What the next arm showed

`none1` ran 07:54–08:13, after the host work stopped. The rep-by-rep structure is the same:

| arm | c=4 rep0 / rep1 / rep2 (tok/s agg) | c=1 rep0 / rep1 / rep2 / rep3 |
| --- | --- | --- |
| piece1 (during the host work) | **64.1** / 37.5 / 37.1 | 18.2 / **16.2** / 19.8 / 21.6 |
| none1 (after it stopped) | **59.2** / 37.6 / 37.9 | 19.1 / **16.8** / 19.9 / 21.4 |

rep=0 fast, reps 1–2 settling near 37.5 at c=4; the same dip at c=1 rep=1; the same rise to rep=3.
A perturbation that happened only during piece1 cannot reproduce itself in none1.

## What it actually is, and what it means for the finding

**Reps inside one server start are not exchangeable.** rep=0 runs against a fresh server and a cold
prefix cache; later reps run against a different cache state. This is the same effect as the TTFT
median that hid a cold prefill (`median=0.60s` while `all=2.93 0.60 0.60`) — the mechanism is
prefix caching, and it is systematic, not noise.

**Consequence for the cgnone2 table: compare like reps across arms, never a mean over reps.** An
arm's rep=0 belongs with the other arms' rep=0. Collapsing reps to a mean mixes a cold and two warm
measurements in a fixed 1:2 ratio and would produce a stable-looking number with no meaning.

## What is still open

`piece1 c=16 rep=1` reported `garbage 1/16`; `none1` c=16 is `garbage 0/16` on both reps. One
occurrence, no reproduction yet — keep it flagged, attribute it to nothing until it recurs in a
later arm.

## The standing rule is unchanged

Do not run heavy host IO or memory work while a benchmark arm is running — one memory pool, and it
did trip the guard. That rule was right; using it to condemn an arm without checking the next one
was not. Two corrections of my own inference in one morning, both from asserting a cause before
looking at the control that was already being measured.
