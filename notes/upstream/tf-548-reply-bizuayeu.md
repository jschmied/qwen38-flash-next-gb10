POSTED 2026-10-10 (user: "check and post answer"). https://github.com/ashhart/TensorFold/issues/548#issuecomment-6094821668

@Bizuayeu thank you, that is far more than a smoke test: real GLM tensors as stored, the cross-check against your own
port, and the three paths our harness skips.

On the prompt-row form for grouped NVFP4 experts: take yours. Nemotron's 4-bit experts already keep a separate prompt
kernel (`experts_prefill.cu`, 64 pairs x 128 columns a CTA, next to the decode `experts.cu`), so a staged prompt form
is the pattern main already has, and yours is 2.8-3.2x faster on full chunks. Our experts PR will keep `experts.cu`
for decode rows only and leave the prompt-row entry to your `experts_prompt.cu` PR, with its oracle fixture and
speed receipt.

The 2 GB read limit in `fixture.zig` we'll raise in our NVFP4 PR. The first step of the interface is up as #587
(Nemotron's 4-bit projections and #482's FP8 behind one weight view); NVFP4 goes behind it next.
