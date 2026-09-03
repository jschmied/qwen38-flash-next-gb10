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

67. **The blockwise FP8 GEMM on sm_121 has a steep M-dependence, and cuBLASLt per-tensor FP8 runs 1.8–2.7×
    faster at large M.** `fp8bench` (preview build's `_C.cutlass_scaled_mm` with 128×128 weight / 1×128
    activation scales, vs BF16 `torch.matmul` (cuBLAS nvjet), vs per-tensor FP8 `torch._scaled_mm`
    (cuBLASLt); 5 × 20 launches, median; `notes/data/fp8bench.txt`). In TFLOPS, in_proj 16,384×2,560:

    | M | blockwise (config) | BF16 | FP8 per-tensor |
    | --- | --- | --- | --- |
    | 64 | 26 (swap 128×32) | 14 | 26 |
    | 256 | 90 (pingpong 64×128) | 47 | 89 |
    | 2,048 | **161** (128×128) | 101 | 162 |
    | 4,096 | **163** | 92 | 172 |
    | 7,503 (odd) | **13.9** (swap 128×32) | 95 | 175 |
    | 8,192 | 95 (128×128) | 100 | 178 |
    | 16,384 | 51 (128×128) | 103 | 176 |

    out_proj (2,560×6,144) and q_proj (12,288×2,560) behave the same (odd M → 15 TFLOPS; 8,192 →
    86–179; 16,384 → 53–179). Three conclusions. (a) The `M % 4` misroute is exactly the 45 ms we saw
    (7,503: 45.3 ms here vs 45 ms in the trace) — 3.3× slower than the aligned 8,192 call for fewer
    rows. (b) Even correctly routed, the 128×128 config **loses efficiency above 4,096 rows** (163 →
    95 → 51 TFLOPS): for FP8-projection checkpoints on sm_121 the right prefill batch is **4,096 or
    2,048, not 8,192** — the missing explanation for the batch-size note (finding 62's 3.1 s at 4096
    vs 4.6 s at 8192 was both effects at once). (c) cuBLASLt's per-tensor FP8 path is flat at
    **~175 TFLOPS** from 2k to 16k — 1.1× the blockwise kernel at 4,096, 1.85× at 8,192, 3.4× at
    16,384. Per-tensor scaling is a different quantisation (quality cost), so the lever is a
    cuBLASLt **blockwise** path on sm_121 (`torch._scaled_mm` with 1×128/128×128 scales, if the
    library supports it here) — `fp8bench2`, queued. If it does, vLLM's `CutlassFp8BlockScaledMMKernel`
    is the wrong default for GB10 at every prefill size, and that is an upstream-shaped finding.

68. **Hyper-connection GEMMs are near their ceiling; lever 5 downgraded.** `hcbench`
    (`notes/data/hcbench.txt`): down 10,240→324 runs at 52–55 TFLOPS BF16 (Triton 55–58), up 320→10,240
    at 43–46 (Triton 47–48, per-tensor FP8 54–59) at M = 4k…32k. cuBLAS picks
    `cutlass_80_tensorop_bf16_s16816gemm_relu_128x256` for the up GEMM. In the 8k trace the two GEMM
    families are ≈ 5 % of prefill; `_hc_combine_norm` + `_hc_gate_mix` another ≈ 9 %, memory-bound.
    Best case for fusion ≈ 4 % of TTFT. Not a priority.

69. **Padding the prompt to a multiple of 4 tokens: −37 % TTFT at 8k, −9 % at 30k, on the shipped
    preview build, no code change.** `TTFTPAD` (batch 8192, prefix cache off, no spec, stock kernels;
    two warm-ups, then 3 requests per arm; `notes/data/ttftpad.txt`):

    | prompt tokens | mod 4 | TTFT median (3) |
    | --- | --- | --- |
    | 7,503 / 7,505 / 7,506 / 7,507 / 7,509 | 3 / 1 / 2 / 3 / 1 | 4.52 / 4.53 / 4.53 / 4.54 / 4.57 s |
    | **7,508** | **0** | **2.86 s** |
    | 29,263 / 29,265 / 29,266 / 29,267 / 29,269 | 3 / 1 / 2 / 3 / 1 | 12.92 / 12.60 / 12.60 / 12.59 / 12.60 s |
    | **29,268** | **0** | **11.50 s** |

    Exactly the `swap_ab = (M <= 64) || (M % 4 != 0)` clause (finding 65): at 8k the whole chunk is
    the tail, at 30k only the last 4,687 tokens are. Request-level workaround for any client of the
    preview image: pad the prompt so the scheduled chunk's token count is a multiple of 4 (with
    chunked prefill, that is the *last* chunk: `prompt_tokens % 4 == 0` when the batch size is a
    multiple of 4). Also: the first 30k request in a fresh server took 15.86 s vs 12.8 s warm — the
    profile's 16.3 s (finding 66) was that cold shape. `ttftpad2` repeats the grid at batch 4096.

70. **Same grid at batch 4096: padding is worth −8 % at 8k and nothing at 30k; batch 4096 + padding is
    the best preview-build configuration.** `TTFTPAD2` (`notes/data/ttftpad2.txt`), medians of 3:

    | | 8k, mod 4 ≠ 0 | 8k, mod 4 = 0 | 30k, mod 4 ≠ 0 | 30k, mod 4 = 0 |
    | --- | --- | --- | --- | --- |
    | batch 8192 | 4.52–4.57 s | **2.86 s** | 12.59–12.92 s | **11.50 s** |
    | batch 4096 | 3.05–3.07 s | **2.81 s** | 11.11–11.16 s | 11.14 s |

    At batch 4096 the misrouted tail is 3,407 tokens at 8k (≈ 0.25 s) and 595 at 30k (nothing). With
    the padding, batch 8192 and 4096 are within 2 % at 8k and batch 4096 is 3 % better at 30k —
    finding 67(b): the 128×128 config is more efficient per row at ≤ 4,096 rows. So on the preview
    build: **keep batch 4096, pad prompts to a multiple of 4** (2.81 s / 11.1 s). Everything beyond
    that needs the main build (dispatch fix + #54513) or a faster blockwise GEMM (finding 67(c)).
