DRAFT — user go given ("ok"). GitHub blazux/qwen3.8-Flash-DGX PR #10 (merged), reply (2026-09-08).

Thanks for verifying it independently before merging — and your case 79 is a better statement of that
bug than mine was.

I had it as "the chunk is sized from the opt-in rather than the opt-in minus the kernel's static
`__shared__`, so 8 of 40 wide-row shapes cannot launch". You have it as a **hard error with a name**,
reproducing on a second box: `chunk_size 256 smaller than TopK 512`. That is the difference between a
latent sizing mistake and a crash your image would have hit at those widths, and it is now confirmed
on hardware that is not mine. I have added your reproduction to the upstream PR's record.

Your micro-bench matches ours closely (1.0–2.4× vs stock here too, decode-sized rows ~1.0–1.1×), and
"a hair above the previous pin" at the model level is what I would expect: the kernel work removed
per-call overhead rather than changing the model path.

One thing worth checking before you re-measure the README figures, because it may move them more than
the kernel did. Your prefill of **2,994 tok/s at 32k** sits almost exactly where the reporter of
[vllm#55397](https://github.com/vllm-project/vllm/issues/55397) measured the *recovered* NVFP4 arm
(2,986 tok/s), which suggests you may already be on the good kernel — but it is worth confirming
rather than assuming, because on sm_121 the default scan stops at
`FlashInferCuteDslNvFp4W4A16LinearKernel`, which dequantizes activations to 16-bit, and never reaches
the native W4A4 kernels below it. The check costs nothing:

```
VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4W4A16LinearKernel
```

Upstream measured −31.6 % prefill from that selection on a dense 27B; [#55405](https://github.com/vllm-project/vllm/pull/55405)
is the fix and is still open. I am running the same A/B on our GB10 right now and will post the
numbers wherever they land — including if they show nothing, which would be worth knowing too.
