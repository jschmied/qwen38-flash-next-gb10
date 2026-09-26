POSTED 2026-09-26 as https://github.com/bilikaz/qwen38-flash-next-recipe/issues/1 — the user's go ("post to mia and bilikaz", 2026-09-26). New issue on bilikaz/qwen38-flash-next-recipe.

Title: RecoverSSM for the GDN layers: frees the per-draft recurrent-state blocks (your ~36k tokens per seat) — measured on one GB10

Your README notes that each running request holds ~36k tokens of pool for the model's recurrent state at K=5. We
traced where that goes on our box, and it is worth checking whether the same fix helps your K=5 setup.

**Where the cost comes from.** In speculative decode, the native GDN path:
- reserves `1 + K` Mamba blocks per request in `align` mode, one per draft position;
- writes the full fp32 state after every one of the K+1 verify tokens (12 MiB per layer per step at our K=3, so
  18 MiB at your K=5), and keeps one of the copies.

Those writes also sit dirty in the 24 MiB L2. The next GEMMs pay the write-back: GDN `out_proj` takes 101 µs in the
trace against 71 µs clean.

**What we did.** We ported the RecoverSSM protocol, already merged in vLLM for Kimi-K3's KDA layers, to the Qwen GDN
layers (and the Qwen4Exp PLE short conv).
- The verify runs from one checkpoint and keeps only a small per-token record (delta-rule correction, key, decay).
- After sampling, one commit replays the accepted tokens and writes the fp32 state once, plus the block-boundary
  state in `align` mode.
- There are no per-draft blocks.

**Measured** on one GB10, vLLM nightly `1ea7c63f4`, MTP K=3, `align` + prefix caching, KV fixed at 4 GiB, 2 server
starts per arm:

| | native GDN spec path | RecoverSSM |
|---|---|---|
| c=1, ms per verify cycle | 55.71–56.65 | 54.37–54.38 (−2.4…−4.0 %) |
| c=4, ms per verify cycle | 105.07–105.41 | 99.59–99.91 (−5.2 %) |
| agent loop (8 dependent turns), s/turn | 1.31 | 1.10 (−16 %) |
| prefix-cache hit rate | 77.4 % | 82.1 % |
| KV tokens in the same 4 GiB | 75,678 | 103,953 (+37 %) |
| GDN `out_proj` (nsys) | 101.4 µs | 73.8 µs |

- **At K=5 both effects should be larger:** 6 blocks and 6 state copies per request per layer instead of 4. We have
  not run K=5 with it. The verify kernel takes any window length.
- **Output:** the state stays fp32. Text differs from the native path by the size of a summation-order change
  (first greedy divergence at a median of 26 tokens, against 31 for a pure reduction-order change); it is
  reproducible across restarts.
- **Requirements:** vLLM's V2 model runner (the merged `RecoverSSMState` commit hook lives there), and PIECEWISE
  CUDA graphs (the builder refuses FULL decode graphs). You already run PIECEWISE.

**Code.** It is a local overlay on a nightly, not a branch yet. There are four patch scripts plus three modules, with
the install order in the README:
- [README](https://github.com/jschmied/qwen38-flash-next-gb10/blob/2e1f7a1b0c35e18fd121849df121fd2ef2b7546d/tools/rssm/README.md);
- [kernels](https://github.com/jschmied/qwen38-flash-next-gb10/blob/2e1f7a1b0c35e18fd121849df121fd2ef2b7546d/tools/rssm/recoverssm_gdn.py);
- [measurements and nsys comparison](https://github.com/jschmied/qwen38-flash-next-gb10/blob/2e1f7a1b0c35e18fd121849df121fd2ef2b7546d/notes/speed-of-light.md#L661-L729);
- we also posted the numbers upstream as a data point for GDN spec decode with prefix caching:
  [vllm#56466](https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579).

Two things from your recipe helped us too: the capture sizes in multiples of K+1 (our own list left c ≥ 3 without
graphs; the A/B is running) and the compaction sysctl, which we are measuring now.

*Measured and written with AI assistance (Claude); every number is from the linked data.*
