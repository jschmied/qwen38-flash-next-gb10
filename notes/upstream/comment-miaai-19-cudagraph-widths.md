DRAFT — needs the user's go. GitHub MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue #19 (2026-09-08).

Owed reply. We said we would test the wide-capture-size arm on our single GB10 and report back. We
did test it, twice, and both runs were **uninformative** — the reason is worth more than the numbers.

**Our arms measured c ∈ {1, 4, 16} only.** With MTP 3 that is 4, 16 and 64 query tokens per step,
and 4 and 16 are captured widths in *both* the default list and the wide list, so the two arms were
running the same graphs at every point we sampled. Predictably null, and null for a reason that has
nothing to do with the hypothesis. Your own commit message states the mechanism we should have
sampled: vLLM's default list leaves decode keys {4, 8, 16} at MTP 3, so a 12-token batch pads to 16
and a 20-token batch decodes eager. **The effect lives at the widths between the captured ones** —
c = 3, 5 and 6 — and your ~4-5 ms on the 5-sequence step is exactly there. We will re-run at those
concurrencies rather than repeat the null.

One arm of ours that did answer something: cudagraph *mode* is not a lever on this box. PIECEWISE vs
FULL_AND_PIECEWISE vs FULL_DECODE_ONLY measured equal in a three-arm A/B (our finding 136), so the
+24 % you saw on the PIECEWISE fallback is worth reading as a symptom of the V1/V2 draft-config
sharing bug you root-caused, not of PIECEWISE itself. Your `VLLM_USE_V2_MODEL_RUNNER=1` pin looks
like the right fix for the right reason.

**On the `MAX_NUM_SEQS` half of this issue, a pointer rather than a claim.** There is an active
upstream thread — [vllm#55533](https://github.com/vllm-project/vllm/issues/55533) — reporting that on
hybrid GDN + MTP the scheduler runs only ~3 of N sequences whatever N is set to, because
`MambaSpec.num_speculative_blocks = num_speculative_tokens` and align-mode `MambaManager` charges
`1 + k` blocks per Mamba group *for the request's lifetime*; the window is
`floor((num_gpu_blocks − 1) / (1 + G·(1+k)))`. If that is what is happening on your box too, then
raising `MAX_NUM_SEQS` cannot flatten the curve and a PLE prewarm will not either — the ceiling is in
the block accounting, not in warmup. We are measuring it here tonight (polling
`num_requests_running` at c=8, with a no-spec arm as the control) and will report the number either
way. `vllm:num_requests_running` against `vllm:num_requests_waiting` in your own `/metrics` would
answer it on your box in one sweep.

Separately, thank you for the `mamba_ssm_cache_dtype=bfloat16` result — it corrected a wrong entry in
our notes. We had recorded that the only route to a smaller attention block was a padding patch,
because fp8 is not a valid `MambaDType`; `FUSED_GDN_STATE_DTYPES` accepting bfloat16 is the route we
missed. Our block is 1,600 against your 3,200 for one reason only: you run fp8 KV and we run bf16,
and the block formula divides the mamba page by the KV bytes per token — so on a hybrid, fp8 KV
*doubles* the prefix-cache block. Your 3,200 → 1,664 is those two effects cancelling.
