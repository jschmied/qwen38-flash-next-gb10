# Why 27% of decode time did not respond to anything

Three interventions on the hyper-connections all measured null. The layers were correctly
ranked and the GPU was busy, so the nulls needed an explanation. Here it is, measured per shape
(`tools/shapebench.py`).

## The measurement

FP8 `_scaled_mm` against BF16 `F.linear`, at the shapes and batch sizes decode actually uses.
L2 defeated by rotating over ~300 MB of distinct weights; every timing printed next to its
roofline (`N*K*2 / 273 GB/s`) so a cache-resident lie is visible rather than believed.

| shape | role | M=1 BF16 | M=1 FP8 | speedup | roofline |
| --- | --- | --- | --- | --- | --- |
| (10240, 320) | hyper-connection **up** | 30.8 us | 26.1 us | **1.18x** | 24.0 us |
| (336, 10240) | hyper-connection **down** | 37.0 us | 41.5 us | **0.89x** | 25.2 us |
| (10240, 2560) | GDN in_proj (control) | 307 us | 126.7 us | **2.42x** | 192 us |

Flat across M = 1, 2, 4, 8 in every case.

## The answer

**These GEMMs are latency-bound, not bandwidth-bound.** At 30–37 us against a 24–25 us
roofline they already run at ~78% of peak for their size, and they do not move with precision.
The control shape — where FP8 genuinely helps — has room (307 us against a 192 us roofline) and
delivers 1.8–2.4x.

They are 27% of decode GPU time because there are **~102,000 of them**, not because any one is
expensive: roughly 200 per forward (48 layers x 2 hyper-connections x {down, up}), ~39 us each
in the live trace, which matches the microbenchmark.

**And the two halves cancel.** FP8 saves ~4 us/call on `up` and *loses* 6–9 us/call on `down`.
Our up-only variant should have bought ~1.3% of GPU time; we measured −1.1% ± 3.0%. The nulls
were the correct answer, not a failed measurement.

## What this closes, and what it opens

Closed: precision and kernel choice on these layers. Blockwise FP8, per-channel FP8, and the
CUTE-DSL skinny GEMM are all attacking a bandwidth problem that does not exist here. Do not
retry them, and do not read the 27% as a bandwidth opportunity.

Open, and the only remaining lever: **the launch count.** ~200 small GEMMs per forward at a
~30 us floor is ~6 ms per forward that is structural. Fixing it means fusing or eliminating
launches, not making each one cheaper — e.g. batching the two hyper-connections within a layer,
or a fused kernel for the whole mHC residual (b12x ships one, `norm.mhc`, but it is
DeepSeek-style and restricted to hidden sizes 4096/7168; ours is 2560, so it does not apply).

## Method note

This microbenchmark should have come first. It costs about two minutes with the server stopped
and it would have pre-empted a checkpoint build, four failed server starts, and three six-run
A/B arms. Rank by profile, then **verify the lever is real at the shape level before building
anything.**

## Addendum: cudagraph capture sizes — a gate we set ourselves, also null

`serve-fnext.sh` pins `cudagraph_capture_sizes` to `[1,2,4,8]`; vLLM would have defaulted our
config to `min(max_num_seqs * (1+k) * 2, 512)` = **96**. A c=16 decode batch is
`16 * (1+2)` = **48 tokens**, so c=16 decode had never run as a captured graph, while c=1
(batch 3) always had.

Raised to `[1,2,4,8,16,24,32,48]`, verified applied (`max_cudagraph_capture_size: 48`, 8 graphs
captured instead of 4):

| | new | control | delta |
| --- | --- | --- | --- |
| c=16 aggregate | 99.2 | 100.1 | −0.8% |
| c=1 decode | 35.43 | 36.45 | −2.8% (inside the control's own sd of 1.04) |

**Null, and predictable from our own byte analysis.** Capturing graphs removes kernel-launch
overhead, which only pays when latency-bound. At c=16 the experts are ~80% of per-step bytes —
each sequence pulls its own ten of 512 while the dense path amortizes — so the regime is
bandwidth-bound and there is no launch overhead left to remove. At c=1 the batch was already
captured, so there was nothing to gain there either.

Reverted: 8 graphs pushed memory to 120/121 GB with 0 available, for no return.

**The prediction was available before the experiment** and would have saved a ~40-minute run.
Both regimes were already characterised: c=1 latency-bound and already captured, c=16
bandwidth-bound. Deriving the expected effect from existing measurements before spending a run is
the cheaper half of this discipline, and it is the half we keep skipping.

## Addendum 2: the FP8 lm_head under speculation — measured, and it compounds

The `lm_head` FP8 gain was originally measured with **speculation off** (+10.1%, 26.1 vs 23.7).
Production runs MTP k=2, and the drafter has **no head of its own** — `mtp.shared_head.head.`
remaps to `lm_head.`, so every draft token is sampled through the quantized head. That raises a
question the original measurement could not answer: does a degraded head spoil drafts?

Matched A/B, same code, MTP k=2, c=1, i4000/o512:

| | fp8mix (BF16 head) | fp8head (FP8 head) |
| --- | --- | --- |
| decode tok/s | 30.60 (n=4) | **36.45** (n=6, sd 1.04) |
| acceptance | **60.3%** (1.207/draft) | 56.6% (1.133/draft) |

**Both effects are real and they point opposite ways.** The FP8 head does cost draft quality —
3.7 pp of acceptance, 6.1% relative. It wins **+19.1%** regardless.

**And the gain nearly doubles under speculation: +10.1% off, +19.1% on.** That is the compounding
argument confirmed by measurement rather than asserted — the head is read once per *draft* token
as well as once per step, so quantizing it saves more when more draft tokens are produced, and
that outweighs the drafts it spoils.

Two consequences:

- The published "+11%" must be qualified as **speculation-off**. With MTP k=2 the figure is
  **+19.1%**, which is the number that matters for how we actually serve.
- It does **not** explain our acceptance sitting below YSLAB's 73.7% at the same k: even with a
  BF16 head we reach only 60.3%. That gap is workload or configuration, and our bench is
  deliberately hostile (unpredictable continuation of random real-corpus slices).

Qwen's own card gives **no recommended k** — only "MTP: 1 layer, trained with multi-steps",
matching `mtp_num_hidden_layers: 1` and the single `mtp.layers.0` in the weights. So k>1 reuses
one layer autoregressively. k=2 as the optimum is confirmed three independent ways: the
architecture, our own k=3 arm, and YSLAB's 0–10 sweep.
