DRAFT — user go given ("go"). GitHub Radar105/qwen38-flash-next-nvfp4-spark, new issue (2026-09-07).

TITLE: sm_121 selects the W4A16 NVFP4 linear kernel — likely ~30% of your prefill, one env var to test

BODY:
Your pinned base `7fbd44cb` (2026-09-05) contains a kernel-selection bug that costs prefill on GB10,
and at 262K context prefill is most of your wall clock — your own headline is a 250K cold prefill of
157.90 s. One environment variable tests it; no patch, no rebuild.

**What happens.** `_POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]` is scanned first-match-wins:

```
1  FlashInferCuteDslNvFp4LinearKernel        gated to sm_10x  -> rejected on GB10
2  FlashInferCuteDslNvFp4W4A16LinearKernel   gate: cc in (100,103) or 120 <= cc < 130
3  FlashInferCutlassNvFp4LinearKernel        native W4A4, never reached
4  FlashInferB12xNvFp4LinearKernel           never reached
5  CutlassNvFp4LinearKernel                  never reached
```

GB10 is sm_121, so the scan stops at #2 — which **dequantizes activations to 16-bit**. A W4A4
checkpoint silently gets an A16 path. I verified the pick on my own GB10 by calling `is_supported(121)`
down the list rather than reading the source; first match is the W4A16 kernel. I also checked your
pinned revision: `7fbd44cb` has the unfixed ordering.

**Upstream:** [#55397](https://github.com/vllm-project/vllm/issues/55397) reports it, measured on a
dense 27B on GB10 as **−31.6% prefill** at 2048 tokens and −19 to −24% decode under load;
[#55405](https://github.com/vllm-project/vllm/pull/55405) is the fix and is still open.

**The test, on your existing build:**

```
VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4W4A16LinearKernel
```

That pushes the scan to `FlashInferCutlassNvFp4LinearKernel` — the arm #55405 makes default, and the
one the reporter recovered +27.5% prefill with.

**Why I think it is worth your time specifically.** Your measured 1,863 and 2,051 tok/s prefill sit
where the degraded arm sits in the upstream repro (2,041 tok/s); the recovered arm there is 2,986. If
that carries, your 157.90 s cold prefill goes to roughly 110 s. I have not measured it on your
configuration, so treat the number as a hypothesis and the mechanism as the checkable part.

**One thing that does not apply to you, so you can skip it.** I have a PLE fix for a semaphore that
runs one step behind, but it is in `PleOffloadConnector` (the #53899 CPU-offload worker) and only
manifests with cudagraphs. You use `VLLM_QWEN4_PLE_MMAP=1` and `--enforce-eager`, so neither
condition holds — mentioning it only so it does not look like something you are missing. You already
carry #55375, which is the PLE fix that does matter.

Separately, and unrelated to speed: your stack does not include
[#55122](https://github.com/vllm-project/vllm/pull/55122). The QSA top-k selection is non-deterministic
on sm_121 — identical requests at temperature 0 can produce different completions, because the sparse
indexer hands out output slots by thread arrival. On my GB10 the unmodified kernel fails to reproduce
its own output on 56 of 56 shapes. It does not affect throughput and may not matter for your use, but
it is worth knowing if you ever rely on repeatability.

Happy to be wrong about the prefill number — if you run the env var and it does nothing, that is
useful to me too, since I am about to test the same thing here.
