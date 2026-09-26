POSTED 2026-09-26 as https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579 — the user's go ("post comment and show url", 2026-09-26). vllm-project/vllm PR #56466 comment (cc #49887).

**Data point for GDN spec decode + prefix caching: the merged RecoverSSM path, ported to Qwen GDN, measured on DGX Spark**

We needed GDN speculative decode with prefix caching for Qwen3.8-Flash-Next (Qwen4Exp, `QwenGatedDeltaNet`) and took a different route than this PR: the RecoverSSM protocol already merged for Kimi-K3 KDA.
- The verify runs from the checkpoint state and stores a per-token record (delta-rule correction, k, g).
- After sampling, the V2 runner's existing `RecoverSSMState` hook replays the accepted tokens and writes the fp32 state once, plus the boundary state in `align` mode.
- It runs with `mamba_cache_mode=align` and prefix caching on the V2 runner, and reuses K3's commit-plan and conv-compaction kernels.
- About 430 lines of kernels and GDN backend, plus small hooks; Qwen4Exp's PLE short conv needed ~150 more.

**GB10 (sm_121), TP=1, nightly `1ea7c63f4`, MTP n=3, align + prefix caching, KV 4 GiB, 2 starts per arm:**

| | native GDN spec path | RecoverSSM |
|---|---|---|
| c=1, ms per verify cycle | 55.71–56.65 | 54.37–54.38 (−2.4…−4.0 %) |
| c=4, ms per verify cycle | 105.07–105.41 | 99.59–99.91 (−5.2 %) |
| agent loop (8 dependent turns), s/turn | 1.31 | 1.10 (−16 %) |
| prefix-cache hit rate (same workload) | 77.4 % | 82.1 % |
| KV tokens in the same 4 GiB | 75,678 | 103,953 (+37 %) |

- **Correctness:** greedy outputs first diverge from the native path at a median of 26 tokens (8 prompts × 512). That is the size of a reduction-order change on this box (31 tokens); the state stays fp32 throughout. Reruns that hit the prefix cache reproduce their hashes (8/8) in every start.
- **Why it helps on unified memory:** in spec decode the native kernel stores the full fp32 state after each of the 4 verify tokens, 12 MiB per layer per step.
  - Those writes sit dirty in the 24 MiB L2, and the next GEMMs pay the write-back: GDN `out_proj` takes 101 µs against 71 µs clean (nsys node trace, reproduced standalone by dirty-L2 sweeps).
  - With RecoverSSM it takes 74 µs.
  - The agent-loop gain also comes from dropping the per-draft Mamba blocks (the hit rate and KV capacity above).
- **Remaining cost:** the commit reads and writes the whole state at the DRAM floor (224 GB/s, ~1 ms/step). Folding it into the next verify, as ReplaySSM does, removes one full read; that variant is being measured now.

**Question:** would a RecoverSSM-based GDN path, reusing the merged K3 infrastructure, be welcome as its own PR? Or should these numbers feed the ReplaySSM work here and in #49887 instead?

- Code, a local overlay, not yet a branch: [kernels](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/tools/rssm/recoverssm_gdn.py#L25-L300), [GDN metadata builder](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/tools/rssm/gdn_recoverssm.py#L60-L129), [README](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/tools/rssm/README.md).
- Measurements: [A/B + trace write-up](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/notes/speed-of-light.md#L655-L722), [raw A/B](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/notes/data/rssm/rssm2b.txt), [trace comparison](https://github.com/jschmied/qwen38-flash-next-gb10/blob/6abd4997c4cedb501c11f2ead280afcf688b74f5/notes/data/rssm/nsys0926-cmp.txt).

*Measured and written with AI assistance (Claude); numbers checked against the linked data.*
