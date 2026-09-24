# NVFP4 draft-head slice — hypothesis, written 2026-09-24 before any run

**Change.** The 32k-row draft-vocab slice of the FP8 lm_head, read by the MTP drafter at every draft step (3 per
verify cycle at n=3), goes from an exact BF16 dequant (160 MiB) to NVFP4 (E2M1, E4M3 scale per 16, FP32 global;
40 + 5 MiB). A Triton W4A16 kernel (BF16 activations) replaces `F.linear`. Plain-max scaling.

**Quality.** The target verifies every draft, so output tokens cannot change through the head's values. Only
acceptance can move. Weight rel-RMSE ~9.5 % (as measured earlier for NVFP4-max on this head), so some draft
argmaxes flip.

**Expected.** An outcome outside these ranges means I debug the instrument first.

| cell | expected |
|---|---|
| kernel vs torch reference (same weights) | max rel ≤ 1e-3, argmax 100 % |
| kernel time vs BF16 `F.linear`, M=1..4, CUDA graph | 0.30–0.60× (bytes 0.28×; the kernel won't hit peak BW) |
| draft argmax agreement vs BF16 slice, random x | ≥ 0.95 (informational; random x is not the drafter's distribution) |
| server acceptance length, c=1 | −0 … −3 % relative (2.525 → ≥ 2.45) |
| server decode ms/tok, c=1 | −0.5 … −3 % (≈1.4 ms saved per ~61 ms cycle, minus any acceptance loss) |
| c=4 aggregate tok/s | +0 … +3 % |
| output hashes vs base | identical at c=1 if the verify batches keep the same numerics. Acceptance changes the verify batch shapes, so a divergence is possible and would NOT be a quality regression; record it either way |

**Decision rule.** Ship only if decode improves beyond both arms' spread, and acceptance does not fall more than the
speed gain.
