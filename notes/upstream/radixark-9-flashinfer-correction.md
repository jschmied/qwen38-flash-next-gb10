DRAFT — needs the user's go. HuggingFace, RadixArk/Qwen3.8-Flash-Next-NVFP4 discussion #9 ("Recipe for vllm"), 2026-09-11.

Thread state checked 2026-09-11 19:1x: one event, our own 2026-09-01 post linking the repo and
REPRODUCE.md. No replies. So this is a correction to our own pointer, not a reply to anyone.

---

## Correction: take FlashInfer 0.6.18, not 0.6.17

The recipe I linked above told you to pin FlashInfer **0.6.17**. That was wrong, and if you
followed it you are missing a kernel. Corrected in the repo today; flagging it here because this
thread is how people found it.

**Why I wrote it.** flashinfer#4757 removed SM121a from the cu130/cu134 aarch64 JIT-cache
architecture lists, and was cherry-picked into 0.6.18. I read that as "0.6.18 loses the prebuilt
cubins on GB10, falls back to runtime JIT, and the ninja fan-out can OOM the box" — the last part
having actually happened here once. I wrote it from the PR description and never opened the wheel.

**What is actually in the two wheels**, `flashinfer_jit_cache/` as installed:

| | 0.6.17 | 0.6.18.post1 |
|---|---|---|
| files containing `121` in the name | **0** | **0** |
| `fp4_gemm_cutlass_sm120.so` | 17 × `sm_120` ELF | the same 17 × `sm_120` ELF |

Neither ships an sm121 artifact. 0.6.17 predates #4757 and has none either, so there were never any
to lose — nothing in the tree is gated to `sm_121a`. `compute_120f` covers CC 12.0 and 12.1, and
`sm_121a` is needed only for sparse MMA (`mma.sp .kind::mxf4nvf4`), which this model does not use.
Everything we touch is a `*_sm120` module, present and identically targeted in both. The 96 files
0.6.18 drops are all `single_decode_with_kv_cache_*`, which vLLM does not call — it uses the batched
paged wrappers.

Empirically, after cutting over: `~/.cache/flashinfer/0.6.18.post1/121a/` holds **0 modules**, and a
full start log has zero `ninja` / `nvcc` / `Compiling` lines. No JIT fallback, so no OOM exposure.

**What the pin cost.** vllm#55715 (merged 2026-09-08) enables the FlashInfer GDN prefill kernel on
SM12x and states FlashInfer ≥ 0.6.18 as a requirement. Three of every four layers in this model are
GDN, and before that fix `_resolve_gdn_prefill_backend()` covered SM90 and SM10x only — so sm_121
fell through to the Triton/FLA fallback silently. On this box that was **1,216 of 1,216** logged
backend announcements. Anyone on the 0.6.17 pin reads themselves out of the fix. Verify from the log,
not the version — both the main worker and the PLE offload worker should say:

```
Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)
```

**Two smaller corrections in the same update.** The vLLM pin now names what we actually run —
nightly `main` plus our port of #53899, which is still open upstream, so PLE offload is still not
something you get from `main` alone. And the recipe listed four source overlays; it is three now,
because FlashInfer 0.6.18 ships `MoERunner.get_cache_key_extras` with `use_fused_finalize` in the
tuple, which is exactly what our autotune cache-key backport existed to add.

**What the kernel is worth here.** Measured after writing the above — 3 starts per arm, same venv,
only the backport differing, ~6,000-token prompt. Ranges, and the arms do not overlap:

| | warm reps | cold rep 0 (full prefill) |
|---|---|---|
| FlashInfer | 0.558–0.564 s | 2.437–2.482 s |
| Triton/FLA | 0.587–0.594 s | 2.610–2.687 s |
| | **+5.0 %** | **+7.1 %** |

Cold is the bigger win because the prefix cache absorbs most of a warm rep, so less GDN prefill
actually runs; the cold cell is the one comparable to upstream's 7.2 % at ISL 32768.

**Not measured:** 32k context, and decode — this kernel does not touch it. One GB10, sm_121, TP1.

Recipe: https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/REPRODUCE.md
Measurement: det-208 and det-209 in `notes/determinism-investigation.md`.
