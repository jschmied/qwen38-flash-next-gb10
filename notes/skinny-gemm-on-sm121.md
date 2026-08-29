# The CUTE-DSL skinny GEMM on GB10: four blockers, and a null at the end

vLLM ships a tuned CuTe-DSL skinny GEMM for this model
(`models/qwen3_8_flash_next/nvidia/low_latency_gemm.py`) and never runs it on GB10. Our profile
put 25% of GPU time in `cutlass_80_wmma_tensorop_bf16` — Ampere-generation kernels on a Blackwell
part — so this looked like the largest lever on the box. Patch:
`patches/skinny-gemm-sm121-gate.patch`.

## Result: no measurable gain

Matched A/B, both arms from an **empty compile cache**, same checkpoint, MTP k=2, c=1,
input-len 4000 / output-len 512, six runs each:

| arm | decode tok/s | kernel dispatching? |
| --- | --- | --- |
| stock gate | **36.45 ± 1.04** | no — 0 swap lines, 0 dispatch lines (asserted, not assumed) |
| gate widened to (12,1) | **35.92 ± 1.36** | yes — `(336,10240)` at M=1,2,4,8 → KERNEL |

−1.5%, diff −0.53 ± 1.37 (95%). **Not significant**, point estimate slightly negative.

The defensible claim is narrow: **upstream's sm_103-tuned configs, applied unchanged to sm_121,
buy nothing.** That is not "the kernel cannot help" — dolf3131 measured +6.9% end-to-end with
configs they swept per shape themselves, and we cover only the fused down projection
`(336, 10240)`; the up projection `(10240, 320)` has no plan entry at all.

## Four blockers, each invisible from above

Every one presented identically — a null inside the 6.9% noise floor. Stopping at any of the
first three would have produced a confident, wrong "the skinny GEMM does not help on GB10".

**1. The arch gate.** `enable_qwen38next_low_latency_gemm()` returns early unless
`_is_sm103()`. Merged PR #53534 states outright that "the CuTe DSL skinny GEMM has no arch
gate", and the gate has been widened to SM100 and SM90 in the last week. The sm_121 case is
unexamined, not rejected.

**2. The plan table is keyed by exact M.** `plan.get(x.shape[0])` — a miss falls through to
`F.linear` silently. Stock keys are `[1, 2, 4, 8]`.

**3. Under PIECEWISE cudagraphs the op's Python body only runs at capture.** Replay bypasses it
entirely, so a call-count-threshold log never fires, and the dispatch decision is frozen at
capture time. Corollary: the M that matters is the **cudagraph capture size**, not the real token
count. We measured decode M ∈ {1,2,4,8} — batches are padded — so an M=3 entry we added for
"MTP k=2 → 1+k tokens" was aimed at a number that never occurs.

**4. A stale compiled graph replayed our pre-patch code.** We patched `_is_sm103()` in
site-packages; the previously-compiled graph was reused and the patch did nothing. Three six-run
arms — stock, gate-only, gate+M=3 — agreed to within 0.2% because they were **the same cached
graph**, not three arms.

⚠️ **Originally written up here as a vLLM defect. It is not, and we verified that before filing
anything.** `KernelConfig.compute_hash()` *is* wired into the cache key at
`vllm/config/vllm.py:506`, so `--moe-backend`, `--linear-backend` and friends **do** invalidate
the compiled artifact correctly. What is not hashed — and cannot be — is an edit to vLLM's own
source. That is expected behaviour, not a bug, and it is unlike
`[[spec-compile-cache-key-omits-nspec]]`, where a genuine *config field* was missing from the
hash.

**The operational rule stands even though the bug does not:** benchmarking a kernel patch made by
editing site-packages requires an empty `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`. Note
`serve-fnext.sh` *unconditionally exports* both, so a systemd `Environment=` override is ignored —
clear the directories instead.

## How to prove a kernel actually ran

Counting swapped layers at load time is not enough — that only proves `quant_method` was
replaced. Log each distinct `(weight shape, M, outcome)` on **first sight** inside the op, where
outcome is KERNEL / no-config / runtime-reject. First-sight logging is essential: a threshold on
call count never fires under cudagraph replay.

## Next

`local-inference-lab/b12x` (`pip install b12x`, pure-Python wheel, pins the
`nvidia-cutlass-dsl==4.6.2` we already run) ships `bf16_gemv_small_n`, measured 1.26–3.0× over
cuBLAS **on GB10** at m=1–4. It is the FlashInfer PR #4250 kernel, closed unmerged. It targets
small-N — i.e. the `(10240, 320)` up projection this exercise could not reach.
