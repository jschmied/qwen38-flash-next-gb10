# Prefill investigation — findings (continues the numbering of `determinism-investigation.md`)

65. **8k prefill profile (2026-09-03): 47 % of GPU time is ONE misrouted FP8 GEMM; upstream fixed it on
    08-19.** `PREFPROF`: stock preview build, prefix cache off, batch 8192, no spec, one 7,503-token
    request under the torch profiler (`notes/data/prefprof.txt`, trace in the scratchpad). GPU busy
    99 % of the 4.63 s span (no idle gaps to chase). By component:

    | component | ms | % |
    | --- | --- | --- |
    | FP8 blockwise projections (`cutlass_3x_gemm_fp8_blockwise`, 97 calls) | 2,203 | 48.1 |
    | MoE (NVFP4 experts, activation, expand) | 630 | 13.8 |
    | other (mostly `_hc_combine_norm` 274 ms + BF16 `nvjet_sm121` GEMMs) | 442 | 9.7 |
    | norm / elementwise / act | 432 | 9.4 |
    | QSA indexer + sparse attention (`_qsa_sparse_paged_gqa_splitk` 357 ms) | 387 | 8.5 |
    | GDN chunked scan / conv | 323 | 7.1 |
    | other GEMM | 160 | 3.5 |
    | PLE / embedding gather | 0.3 | 0.0 |

    The 97 FP8 calls are two per layer: the merged GDN/QSA input projection (N=16,384 or 12,288,
    K=2,560) at **~45 ms** and the output projection (N=2,560, K=6,144) at ~2.5 ms — 48 calls × 45 ms
    = 2.0 s of the 4.6 s. Both run the same template: `cutlass_3x_gemm_fp8_blockwise<…, 128, 1, 128,
    tile 128×32×128, …, swap_ab=true>` — the **small-M swap-AB config** that main only uses for
    M ≤ 64. Cause: the preview build's dispatch reads `swap_ab = (M <= 64) || (M % 4 != 0)`; 7,503 is
    odd, so the whole chunk went to the tile-N=32 kernel (14 TFLOPS on the wide projection vs
    94 TFLOPS on the narrow one with the same kernel). Upstream removed the `M % 4` clause in
    **vllm#52775 (2b7fcbf5, 2026-08-19, "SM120: stop routing misaligned-M blockwise FP8 GEMMs to the
    small-M swapAB config")** — after the preview's fork point, so the shipped `_C` still has it.

    This also explains two things we had measured without a cause: finding 62's 3.12 s at batch
    4096 (chunks 4,096 + 3,407: only the tail is misaligned) vs 4.60 s today at batch 8192 (the whole
    7,503 chunk is), and the "prefill batch size is workload-dependent" note. It applies only to
    FP8-projection checkpoints (`fp8head`, our +39 % decode build); RadixArk's BF16 projections use
    the `nvjet_sm121` cuBLAS kernels and are unaffected — which is why every field number sits at
    2.3–2.7k tok/s and ours did too.

    Predictions under test: (a) padding the prompt to a multiple of 4 tokens removes ~2 s at 8k on
    the preview build (`ttftpad`, queued); (b) a main-tree build (nightly aarch64 wheel
    `0.28.1rc1.dev352`, torch 2.13.0, flashinfer 0.6.18) removes it for every length; (c) the 30k
    profile shows the same kernel dominating (29,263 = 3 × 8,192 + 4,687, one misaligned chunk).
    PLE gather: 0.0 % — lever 2 of the plan is dead as a prefill lever. Hyper-connections
    (`_hc_combine_norm` + `_hc_gate_mix` + BF16 GEMMs) ≈ 12 %, second after the FP8 fix.

66. **30k profile confirms the mechanism, and the 30k number itself is warm-up-contaminated.** Same
    server, 29,263 tokens = 3 × 8,192 + 4,687: the three aligned chunks run the FP8 projection in the
    **default 128×128 config at 6.3 ms per 8,192 rows** (288 calls; ~109 TFLOPS on the wide
    projection), the misaligned 4,687-row tail runs the **swap-AB config at 13.5 ms** (100 calls) —
    3.7× slower per row (at 7,503 rows it was 7.8×: the small-M kernel degrades superlinearly with M).
    FP8 total 3.4 s of 13.35 s kernel time; MoE 2.6 s, QSA indexer+attention 1.8 s, GDN 1.4 s, all
    linear in tokens vs 8k. GPU busy only 85 %: 2.0 s of gaps > 5 ms, the largest 1.29 s at 0.5 s into
    the run with driver/attribute queries and kernel launches inside — first-time JIT/autotune for the
    new 4,687-row shape (the warm-ups were 8k only). So the 16.3 s TTFT here is not comparable to
    finding 62's 11.3 s at batch 4096; `ttftpad` (8192) and `ttftpad2` (4096) re-measure both lengths
    warm, 3 requests each, with prompt lengths mod 4 = 0…3. Raw: `notes/data/prefprof.txt`.
