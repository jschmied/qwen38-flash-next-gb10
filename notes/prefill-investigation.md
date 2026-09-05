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

71. **The large-M collapse of scaled FP8 GEMMs on GB10 is an L2-locality effect, not a kernel bug —
    and chunking M at 4,096 rows recovers 3×.** Probes on the idle GPU (in_proj shape 16,384×2,560,
    5 × 10 launches, median):

    | M | blockwise, 1 launch | blockwise, 4,096-row chunks | cuBLASLt row-wise, 1 launch | row-wise, chunks |
    | --- | --- | --- | --- | --- |
    | 8,192 | 7.6 ms (91 TFLOPS) | **4.3 ms (161)** | 7.0 ms (99) | 4.2 ms (165) |
    | 16,384 | 26.2 ms (53) | **8.6 ms (160)** | 26.3 ms (52) | 8.3 ms (165) |
    | 32,768 | 52.9 ms (52) | **17.0 ms (161)** | 52.3 ms (53) | 16.5 ms (166) |

    Two independent libraries (vLLM's CUTLASS 3.x blockwise kernel and cuBLASLt's row-wise path)
    degrade to the same numbers; cuBLASLt's per-tensor path does not (≈ 175 TFLOPS at every M). GB10:
    **48 SMs, 24 MiB L2**; the FP8 weight operand here is 42 MB. A tile raster that walks M-major
    re-streams the weight from DRAM once per tile row as soon as M spans many tile rows — the
    per-tensor kernel evidently rasters/swizzles so the weight stays L2-resident. Also: PyTorch gates
    DeepSeek-style (1×128 / 128×128) `_scaled_mm` to SM90 ("only supported in CUDA for SM90"), so a
    cuBLASLt blockwise path is closed on the torch side regardless.

    Consequences. (a) For sm_121 the right fix is in the tile scheduler (CUTLASS `RasterOrder` /
    `max_swizzle_size`) or, trivially, an M-chunk loop at ≤ 4,096 rows in the blockwise caller; the
    chunk loop is a 5-line Python change with a 3× before/after at 16k. (b) Anyone serving a
    blockwise-FP8 checkpoint on GB10 (Qwen's official `-FP8`, lovedheart's FP8-mixed, crimsonjoo's
    "hybrid") at the default `--max-num-batched-tokens` (8192+) pays 1.7–3× on every FP8 projection in
    prefill; batch 4096 sidesteps it, which is why our prod setting was right without knowing why.
    (c) For our own TTFT the remaining lever is therefore the main build (dispatch fix + #54513), not
    the kernel — at batch 4096 we already sit at the 160-TFLOPS regime. (d) Upstream-shaped: an
    issue with this table, and the chunk-loop or raster-swizzle PR. Not posted; needs the user's go.

72. **INCIDENT — the main-build load thrashed the box; reboot 17:08.** `fnmain1` (nightly dev352 venv,
    body 69.4 GiB on the GPU at `--gpu-memory-utilization 0.60`, the 47.7 GiB PLE tables pinned in host
    memory via `--cpu-offload-gb 56 --cpu-offload-params ngram_embedding`, `--language-model-only`)
    passed model construction (CUTLASS blockwise FP8 kernel selected, Triton GDN, FlashInfer CUTLASS
    MoE — both loader patches accepted) and then sat at 120 of 121 GB during the safetensors read;
    from 16:52 the kernel logged hung tasks (`cache_mgr_main` blocked > 614 s), journald flushed
    caches under memory pressure, the machine stopped responding and was rebooted from the
    workstation at 17:06–17:08. Pinned (unswappable) host memory for the tables plus the GPU
    reservation plus the loader's page cache exceed the pool — the preview's separate offload
    worker held the same 47.7 GiB but not pinned, and the box lived at 119/121 GB. Lost with the
    reboot: the /tmp scratchpad (runners, traces, microbench sources; every result of the day was
    already in `notes/data/`, `mtpgrid0` transcribed to `notes/data/mtpgrid0-partial.md`). Survived:
    `/opt/llm/kernel-det/_C_det.so`, `/opt/llm/serve-fnmain.sh`, the fnmain venv with PLEGATE +
    SCALEINV installed (FP8CHUNK not yet). Rule: on GB10 never pin the PLE tables; a main build
    needs either the CPU-offload worker ported or a compressed PLE (HashK 12.8 GB on the GPU) — the
    latter is now a *memory* lever, not a speed one. The main-build TTFT question stays open.

73. **M-chunking the blockwise GEMM is exact — only with re-materialised scales.** Bit-level check
    (in_proj shape, realistic 1×128 / 128×128 quantisation, fp32 reference): single launch, chunked
    with `As[i:i+c].t().contiguous().t()`, and the reference agree (mean |err| 0.0011 on outputs of
    mean |y| 0.81 = FP8 noise; chunked == single launch in every element at 8k and 16k). A plain
    row-slice of vLLM's **column-major** activation scales (`QuantFP8(column_major_scales=True)`,
    stride (1, M)) is silently wrong: mean error 0.15, ~99 % of elements off — the kernel deduces
    the scale layout from its own M. `tools/main/fp8chunk_patch.py` re-materialises per chunk
    (4,096 × K/128 floats, negligible). Yesterday's microbench timings were unaffected (layout does not
    change the kernel's work), the earlier "chunked != unchunked" was this layout mismatch, not a
    kernel defect.

74. **Main tree serves the FP8-mixed checkpoint on GB10 — via the vllm#53899 offload worker — and the
    misroute is gone by default: TTFT 8k 2.80 s, 30k 10.87 s, flat across prompt residues.** `FNMAIN3`
    (`notes/data/fnmain3.txt`): nightly `0.28.1rc1.dev352` venv + the #53899 overlay/hand-port
    (`tools/main/`) + four loader patches for this checkpoint (PLE gate for `modelopt_mixed`,
    `weight_scale_inv` rank-2 → `weight_scale`, `quant_config` on the body `ParallelLMHead`, a
    block-scale branch in `VocabParallelEmbedding.weight_loader` for the head's `[1940, 20]` scales),
    `--kv-cache-memory-bytes 2 GiB` (the utilisation heuristic counts the offload process's 48 GB and
    comes out at −11 GiB at util 0.80), `--language-model-only`, batch 4096, prefix cache off, no spec.
    Model 69.84 GiB in 508 s; KV 68,056 tokens (2.08× at 32k); up after 660 s; smoke text coherent.

    | | preview, unpadded | preview, padded to 4 | **main, any residue** |
    | --- | --- | --- | --- |
    | 8k (7,503 tok) | 3.05 s | 2.81 s | **2.80–2.81 s** (6 residues, 3 requests each) |
    | 30k (29,263 tok) | 11.11 s | 11.14 s | **10.84–10.90 s** |

    So #52775 is confirmed end to end (main unpadded = preview padded), and main is a further −2 %
    at 30k (the #54513 indexer split, at batch 4096 where the FP8 GEMMs are already in their good
    regime). At batch 4096 the C++ chunk fix (PR #55180) has nothing to add; its case is batch ≥ 8192.
    Memory guard (`/opt/llm/runners/memguard.sh`, PSI-based) never fired; pressure stayed ≤ 0.6 %
    throughout while `available` sat at 2 GiB — the availability-based guard would have aborted a
    working server, which is why it was replaced (the preview always ran like this).

75. **PR #55180's C++ chunk path, standalone on GB10 (same build for both columns, CUTLASS v4.7.1):
    48/48 bit-identical to the unmodified op, 1.66–3.16× on weights above the L2, no point below
    0.98×.** `notes/data/fp8chunk_standalone_v2.txt`. The rule that survived the sweep: ~12 MiB of
    activation per launch (K-aware: 4,096 rows at K=2,560, 2,456 at K=5,120), chunks balanced and
    4-aligned, chunking from 1.5 chunks of rows; gate = weight bytes > L2. The first, fixed-4096
    version (`fp8chunk_standalone_v1.txt`) lost 5–7 % at M=4,097 and left the 5120² weight at
    ~120 TF; the K-aware version is 1.00–1.01 at 4,097 and 159 TF on 5120² from 4k rows up.
    Earlier comparisons against the preview's `_C` overstated odd-M ratios (the preview still has the
    #52775 misroute) and understated even-M ones by 2–7 % (different build) — hence the same-build
    baseline. PR moved out of draft with this table.

77. **GDN chunked delta rule: fla-core 0.5.2 is 6–14 % faster than vLLM's vendored FLA on GB10 at the
    model's prefill shapes, same outputs (bf16 rounding).** Standalone, H=48, K=V=128, one sequence,
    `use_qk_l2norm_in_kernel=False`, 5×10 launches: T=2048 2.47 → 2.33 ms (1.06×); 7,503 8.94 → 8.07
    (1.11×); 16,384 19.8 → 17.9 (1.11×); 29,263 37.4 → 32.7 (1.14×); max |Δ| 0.0078 = bf16 ulp. Per
    layer × 36 GDN layers this is ~31 ms of a 2.8 s TTFT at 8k (≈ 1 %) and ~170 ms of 10.9 s at 30k
    (≈ 1.6 %). Free (a vendored-copy sync upstream), small; the GDN share itself (7–10 %) is the
    ceiling. Not pursued further today.

78. **MoE at prefill: the FlashInfer CUTLASS grouped GEMM has no faster tactic on GB10; the layer
    runs ~2× above its weight-streaming floor.** Standalone `cutlass_fused_moe` with the model's
    geometry (512 experts, 2560→640, top-10, NVFP4 W4A4, random weights) at M=7,503: **13.5 ms per
    MoE layer = 54 TFLOPS all-in** (738 GFLOP); the two grouped GEMMs are 9.3–9.9 ms of it, expand +
    activation ~2.2 ms. Forcing every GEMM1 tactic (0–31) and every GEMM2 tactic (0–63) via
    `profile_ids`: all within 13.2–13.7 ms (±2 %), i.e. the autotuner's noisy per-bucket choices
    (finding: tactic ids jump between neighbouring buckets) are harmless — the tactic table does not
    contain a faster kernel for ~147-row-per-expert problems on sm_121. Floor estimate: expert weights
    1.26 GB/layer at 273 GB/s ≈ 4.6 ms + activation traffic ≈ 2 ms ≈ 7 ms. So ~1.5–1.9× is left in
    principle, but not via tactics: it needs a different kernel family (`--moe-backend` A/B at prefill
    on the main build is the practical test), or fewer/larger per-expert tiles (a kernel change).
    Harness: `tools/moe_tactics.py`.

79. **NVFP4 MoE backend at prefill (main build, offload worker, batch 4096, one start each, 3 requests):
    FlashInfer CUTLASS stays the fastest; nothing else loads or beats it.** `notes/data/moeab.txt`:

    | backend | 8k TTFT | 30k TTFT | note |
    | --- | --- | --- | --- |
    | `flashinfer_cutlass` (default) | **2.86 s** | **10.98 s** | |
    | `marlin` | 2.98 s (+4 %) | 11.59 s (+6 %) | coherent text |
    | `humming` | 2.95 s (+3 %) | 11.48 s (+5 %) | coherent text |
    | `cutlass` (vLLM's own) | — | — | wrong CLI name in the runner (`vllm_cutlass`); known illegal-memory-access at init on this box |
    | `flashinfer_cutedsl` | — | — | refused at init: "kernel does not support current device" (sm_121) |

    With finding 78 (tactic table flat, ~2× above the streaming floor) this closes the "different
    kernel family" route on the current software: the MoE share of prefill (14–19 %) stays where it
    is until FlashInfer's grouped GEMM handles ~150-row expert problems better (fewer/larger tiles).
83. **`chunke2e` C4k arms fail at startup — the Python M-chunking patch is not compile-safe.** Root cause
    (`fnext-C4k_a.log`): `Dynamo does not know how to trace builtin operator print` — v1 resolved the env
    and printed its activation line lazily *inside* `apply_block_scaled_mm`, which torch.compile traces;
    the first forward (KV profiling) aborts. The same trace region would also have specialised the Python
    `for` loop on the symbolic M (the P1 point of the #55180 review, which is why the PR moved the loop
    into the C++ op). v2 (`tools/main/fp8chunk_patch_v2.py`) resolves the env and prints at import and puts
    the loop into an opaque `torch.library.custom_op` (`fp8chunk::scaled_mm_chunked`, fake impl returns
    the [M, N] empty), so compile sees one opaque call guarded on `M > chunk`. The off arms (C0) die too —
    v1 calls the printing helper unconditionally — so this run yields nothing (`notes/data/chunke2e-failed.txt`);
    the venv was reverted cleanly at the end. Redo runner `chunkredo` is staged (waits for the
    whole chain), **not started** — needs the go.

85. **`qsadump` produced no dump (02:07–02:20).** Both probes returned 200, the hook site
    (`qsa_select_paged_tokens`, the only selection path with metadata present) is right, but the patch
    capped itself at 16 dumps counted from process start: the warmup/profiling passes consumed the
    budget, the runner then deleted those files as intended and the real prompts found the counter
    exhausted. v2 (`tools/determinism/qsadump_patch_v2.py`) dumps only while `<dir>/ARM` exists (the
    runner creates it once the server is up), names files by row count, budget 96; the redo runner
    `qsadump2` also runs `--enforce-eager` so no compile cache can serve a stale graph. Staged behind
    the chain, **not started** — needs the go.

86. **FLA shared-memory gate (102400 → 101376) + `chunk_delta_h` `num_warps=2` pin (blazux/Saren-Arterius)
    — no effect on our vendored FLA (`flagate`, `notes/data/flagate.txt`).** `chunk_gated_delta_rule`
    before/after at T = 2048 / 7503 / 16384 / 29263: 2456 → 2486, 9003 → 8932, 19864 → 19953,
    36312 → 36073 µs, i.e. ±1 % with identical numerics (max|diff| 0.0078 in both). fla-core 0.5.2 stays
    8–12 % faster than the vendored copy at every length (finding 77 reproduced: 1.03 / 1.10 / 1.11 /
    1.12). So the 99-KiB gate is not what the hot kernels consult on sm_121, and the warp pin matches
    the autotuner's pick; the swap to fla-core is the lever, not the gate. `tools/main/fla_gb10_patch.py`
    is kept only as a record. Caveat: the runner left the patch installed and the grid redo (`s7redo`)
    started on that venv 02:23; given the ±1 % no-op with identical outputs the redo counts as stock;
    `flarevert` removes the patch after the redo and the #50729 application.

89. **Server-level M-chunking on Flash-Next (`chunkredo`, main venv, batch 8192, v2 patch active on the selected
    `CutlassFp8BlockScaledMMKernel` path, two interleaved starts; `notes/data/chunkredo.txt`): null.** TTFT 8k
    chunk-on 2.73 / 2.72 s vs off 2.71 / 2.71 s; 30k on 10.70 / 10.66 s vs off 10.64 / 10.62 s. Explanation is
    the checkpoint, not the kernel: Flash-Next's largest FP8 blockwise weights are 25 MiB (`in_proj_qkv`, 36×)
    and 30 MiB (`q_proj`, 12×), barely over the 24 MiB L2, and they carry ~20 TFLOP per 8k prefill, so even a
    full recovery to the chunked rate bounds the gain at ~3 % — inside single-start noise. This is the
    server-level number for PR #55180 on this model: "no measurable change, no cost" (the C++ gate would not
    even fire here). The models that show the PR's gain have 100+ MiB FP8 projections (the 27B's are 60–120
    MiB, parked). The venv was reverted after the run (0 patch lines).

90. **MoE tile-boundary hypothesis REFUTED (`moel2`, two starts, `notes/data/moel2.txt`).** The NVFP4 grouped
    GEMM shows no step at any M-tile boundary (64 / 128 / 256 rows per expert): time per token falls
    monotonically, 4.03 µs at M=2048 → 2.47 at 4096 → 1.86 at 7503 → 1.76 at 8192 → 1.38 at 16384, balanced
    and random routing within 2 %. So the kernel does not re-stream expert weights per tile, and larger
    prefill chunks are strictly better for the MoE — the batch-4096 optimum on the preview came from the
    FP8 misroute, not from here. What the sweep does show: the kernel tops out at ~70 TFLOPS on a part
    with ~1 PFLOPS dense FP4, i.e. it is neither bandwidth- nor compute-bound but *shape*-bound — 512
    experts × N=640 give tiny per-expert tiles and low occupancy. That is a kernel-config problem
    (tile shape / split-K / cluster) not an L2 one, and it is the "2× above the floor" of finding 78.
    Retire plan §6 item "moel2"; the MoE lever is a FlashInfer grouped-GEMM config for small-N experts.
91. **Hyper-connection kernels (`hcbench2`, two starts, `notes/data/hcbench2.txt`): one of the two has
    headroom.** Torch's own elementwise floor on this box is 330–340 GB/s (not the 273 GB/s spec).
    `_hc_combine_norm` stock runs at 175–177 GB/s (1.56 ms at 4096 rows, 3.08 ms at 8192); the re-tiled
    v2 (one program per row, all four streams, block output read once) reaches 213–224 GB/s: **×1.21 at
    4096, ×1.26 at 8192**, outputs identical for `out`, y within bf16 rounding (0.0156). `_hc_gate_mix`
    stock is already at 230 GB/s and no tiling beats it (×0.99–1.00). At 8k prefill the combine-norm
    saves ~0.6 ms × 48 layers ≈ 30 ms of 2.71 s (1 %), at 30k ~4 × that (~1 %). Real but small; the
    remaining gap to the 335 GB/s floor is the second pass over `out` and can be closed only by fusing
    the norm into the consumer. Keep as a low-priority patch candidate, not a plan item.
92. **QSA block-selection overlap (`qsadump2`, 96 dumps over the 8k and 30k prefills, `notes/data/qsadump2.txt`):
    the tile-union kernel is a clear GO.** Consecutive-query Jaccard 0.87–0.94 in the 8k prefill's
    3,813-row chunks and 0.4–0.8 elsewhere; the union of selections over 64 consecutive queries is
    487–2,536 blocks against 64 × 374–512 per-row gathers: **gather traffic saved 90–98 % at tile 64,
    94–99 % at tile 128**, worst case (30k, 3,072 visible blocks) still 90 %. Mean |sel| is 374 in the
    first chunk (top-k < 512 because fewer blocks are visible) and 512 after. This is the number plan
    §5 item 3 needed: a kernel that gathers the union once per query tile and masks per row would cut
    the sparse attention's KV traffic by ~20×; whether that converts to time depends on whether
    `_qsa_sparse_paged_gqa_splitk_kernel` (0.36 s at 8k, 1.5 s at 30k) is gather-bound — next step is a
    tile-union prototype of that kernel, standalone, against the dumped selections.

94. **Trailing-block drop under MTP: the flag is worth −26 % per warm agent turn (`blockdrop3`, main build,
    MTP n=3, prefix caching on, batch 4096, three interleaved starts each; `notes/data/blockdrop3.txt`).**
    `disable_eagle_block_drop=true` (#53388) vs default:

    | | default (drop) | flag on |
    | --- | --- | --- |
    | s per turn, 8-turn loop (a / b / c) | 2.75 / 2.49 / 2.51 | 2.15 / 2.11 / 2.10 |
    | warm turns 3–8, mean of 18 | 2.05 s | **1.52 s (−26 %)** |
    | cached tokens per warm turn | 4,800 | **6,400 (+33 %)** |
    | MTP acceptance | 56.1 / 53.3 / 53.8 % | 59.5 / 57.5 / 60.0 % |
    | turn 1 (cold) / turn 2 | 4.65–3.62 / 4.7–4.12 | 3.83–3.61 / 4.16–4.22 |

    Acceptance does not move (the PR's caveat does not bite here); the on-arm's total is also more
    reproducible (2.10–2.15 vs 2.49–2.75). Turn 2 is cold on *both* arms — that is the align mode's
    "first repetition never hits" behaviour (`prefix-cache-align-mode-dead`), not the block drop; the
    flag cannot touch it and it is the next-largest per-turn cost. This is plan §5 item 1 closed:
    the missing evidence for #53670 / #50897 now exists on a hybrid + align + in-checkpoint-MTP
    model, and the fix is a merged one-line flag on main. **Prod implication:** any main-build serve
    with MTP + prefix caching should set it. For the preview image (blazux et al.) it is a 4-file
    port of #53388. Also: this run was the main build's first serve with MTP at all (finding 93).

95. **Drop-in M%4 pad for the preview's blockwise-FP8 GEMM: validated at the server level (`m4pad2`,
    preview venv, batch 8192, prefix cache off, no spec, two interleaved starts; `notes/data/m4pad2.txt`).**
    `tools/main/fp8_m4pad_patch.py` v2 (M test inside the opaque custom op — see `failure-modes.md`):

    | | patch on (a / b) | patch off (a / b) |
    | --- | --- | --- |
    | TTFT 8k, median of 3 | **2.84 / 2.84 s** | 5.03 / 3.51 s |
    | 8k, all requests | 4.58 2.84 2.82 / 3.10 2.84 2.80 | 5.03 6.64 3.51 / 5.22 2.93 3.51 |
    | TTFT 30k, median of 3 | **11.28 / 11.16 s** | 13.52 / 11.72 s |

    With the pad the 8k number is 2.84 s on every non-warm-up request, the same value the request-level
    padding gave (finding 69) and the same as the *fixed* main build (2.71 s, finding 74) within the
    preview/main gap. Without it the stock kernel is not just slower but **bimodal** (2.9–6.6 s at 8k,
    11.5–14.9 s at 30k): the swap_ab path's cost depends on how the scheduler happens to cut the chunk.
    Effect on the shipped image at batch 8192: −40 % TTFT at 8k, −10–15 % at 30k, and batch 8192 is
    strictly better than 4096 again. This supersedes the "batch 4096 + request padding" recommendation
    (finding 70) for anyone on the vendor image: install the patch, run batch 8192. Handed to blazux
    (posting-log item 21). The venv was reverted after the run.

96. **QSA union-kernel pre-test: the sparse-attention loop is 3× faster at M=64 — GO, with a shape
    constraint (`qsablockm2`, two starts, `notes/data/qsablockm2.txt`).** A stripped copy of
    `_qsa_sparse_paged_gqa_splitk_kernel`'s loop (gather K/V by index, QK dot, online softmax, PV dot; synthetic
    data at the model's shapes, 4,096 query rows × 16 heads × 2,048 selected tokens × head_dim 256) with ROWS
    query rows per program sharing one index list, i.e. dot M = 16·ROWS:

    | ROWS | M | tiles / stages | µs per row (a / b) | TFLOPS |
    | --- | --- | --- | --- | --- |
    | 1 (today) | 16 | 64 / 2 | 2.91 / 2.9 | 11.5 |
    | 2 | 32 | 64 / 1 | 1.52 | 22 |
    | 4 | 64 | 64 / 1 | **0.92–0.94** | **36** |
    | 8 | 128 | 32 / 1 | 0.93 | 36 |
    | 8 | 128 | 16 / 1 | 1.48 | 23 |

    M=16 reproduces the real kernel's measured ~12.7 TFLOPS, so the loop is representative. The gain saturates at
    M=64: **GB10's 99 KiB shared memory** holds q[M,256] + K[256,BN] + V[BN,256] in bf16 only up to M=64 at
    BN=64 with one pipeline stage (M=128 needs BN=32; two stages fit only at M≤32), and 16 warps or narrower
    tiles lose it again. Design that follows: **4 consecutive query rows per program, 64-column tiles, one
    stage, 8 warps, per-row masks over the tile's union** — union overhead at 4 rows is ≤ 1.15× columns
    (consecutive-row Jaccard 0.87–0.94 at 8k, finding 92), so ~2.5× net on the kernel ≈ −8 % TTFT at 8k,
    −9 % at 30k. Effort: new Triton kernel + union/mask precompute + integration on the indexer path +
    correctness vs the dumped selections, 2–3 days. Decision pending.

97. **Warm-turn decomposition on a prefix hit: the intercept is the un-hit tail of the 1,600-token align
    block, not a kernel (`hitprobe`, main build, prefix caching on, batch 4096; `notes/data/hitprobe.txt`).**
    Streamed first-token time vs new tokens appended to a cached ~7.6k-token prefix, medians of 3:

    | new tokens | MTP n=3 + flag | no spec |
    | --- | --- | --- |
    | 0 (identical request) | 0.592 s | 0.637 s |
    | 1 | 0.610 | 0.644 |
    | 130 | 0.746 | 0.734 |
    | 1,000 | 0.978 | 0.969 |
    | cold seed / 2nd identical | 3.00 / 3.15 s | 2.96 / 3.04 s |

    Slope 0.34–0.37 ms per new token (= the 2.7k tok/s prefill rate); intercept ~0.6 s independent of the
    drafter. vLLM sets the attention block to **1,600 tokens** ("to ensure that attention page size is >=
    mamba page size", `interface.py:918`); a 7,640-token prompt hits 4 blocks = 6,400 tokens and re-prefills
    the remaining ~1,240 on every warm turn ≈ 0.46 s; the rest (~0.13 s) is fixed. So a 1.52 s warm turn is
    ≈ 0.46 tail re-prefill + 0.13 fixed + 0.05 new-token prefill + ~0.8 decode (no-spec decode measured at
    40.5 ms/token = 24.7 tok/s; MTP ~27 ms/token from the loop totals — the streamed per-delta number
    under MTP is the delta-vs-token trap and is not a token rate). The second identical request re-prefills
    fully on both arms ("first repetition never hits", independent of spec). No supported knob changes the
    granularity: `--mamba-block-size` is overridden in align mode, `MambaDType` has no fp8, and fp8
    attention KV would *double* the block (fewer bytes per token). The fix is upstream RFC #45702 (partial
    cache hits, copy-on-write tail); `hitprobe3` tests the boundary-padding workaround.

98. **Boundary-aligned prefix: the warm-turn intercept collapses from 0.59–0.75 s to 0.15–0.27 s
    (`hitprobe5`, main build, MTP n=3 + flag, `notes/data/hitprobe5.txt`).** The probe pads the shared prefix
    (seed turn + assistant turn + the next user header) so it ends exactly on the 1,600-token align block; the
    chat-template suffix after the new content turned out to be 10 tokens (the S-sweep dips only at S=10 and,
    by coincidence of the second boundary, S=13). On the same server, same prompt length ±5 tokens:

    | | shared prefix mis-aligned (prompt mod 1600 ≈ 1,240) | aligned |
    | --- | --- | --- |
    | hit, +1 new token | 0.61–0.84 s | **0.153 s** |
    | hit, +130 new tokens | 0.75 s | **0.265 s** |
    | identical request, prompt exactly on a boundary | 0.71 s (the last block is always recomputed) | — |

    So the fixed cost of a hit on this stack is ~0.1 s and everything above it in finding 97 was the tail
    re-prefill. Two consequences: (1) **workaround for agents with a static system prompt: pad it so the
    shared prefix is a multiple of 1,600 tokens** (`tools/hitprobe_aligned.py` shows how to find the exact
    pad against a live server; the template suffix is 10 tokens for this model with thinking off) — worth
    0.5–0.6 s per warm turn, i.e. a 1.52 s turn → ~1.0 s; the loop's later turns drift off the boundary as
    the transcript grows, so the average saving over a session is about half of that unless the client
    re-pads; (2) the upstream fix is #45702 (partial cache hits) and this is the per-turn number it lacks.
    Also seen: with the 2 GiB KV pool (23 blocks) the sweep's 24 seeds evicted each other's last block —
    the first request after an eviction pays one block (0.9 s); medians are quoted.

99. **gau-nernst's blockwise-FP8 kernels on GB10 (`gnbench`, two starts, standalone, `notes/data/gnbench.txt`):
    neither beats CUTLASS-in-L2 nor chunking; each has half of the answer.** TFLOPS, start a / b within 3 %:

    | shape, weight | M | CUTLASS single launch | their Triton (swizzle 8) | their CuteDSL sm120 | PR #55180 C++ chunking (finding 75) |
    | --- | --- | --- | --- | --- | --- |
    | in_proj_qkv 10240×2560, 25 MiB | 4k / 8k / 16k | **173** / 90 / 64 | 101 / 101 / 101 | 145 / 98 / 51 | 155–170 at every M |
    | q_proj 12288×2560, 30 MiB | 4k / 8k / 16k | **172** / 97 / 53 | 101 / 101 / 100 | 143 / 104 / 50 | " |
    | 16384×2560, 40 MiB | 4k / 8k / 16k | **164** / 92 / 52 | 101 / 101 / 101 | 139 / 97 / 50 | " |
    | 5120×5120, 25 MiB | 4k / 8k / 16k | 126 / 75 / 72 | 100 / 98 / **102** | 119 / 52 / 52 | " |

    All bit-identical to CUTLASS (max|diff| 0). The Triton kernel's `swizzle2d` raster is what a GB10 kernel
    needs — flat at every M, 1.9× the collapsed CUTLASS at 16k — but its mainloop tops out at ~100 TF (Triton's
    FP8 `tl.dot` on sm_121; `dot_scaled` is worse, 66 TF). The CuteDSL kernel has the mainloop (145 TF at 4k,
    0.84× CUTLASS) but no L2-aware raster, so it collapses exactly like CUTLASS (98 → 51). The kernel that would
    beat chunking is the CuteDSL mainloop with the Triton kernel's swizzle — not built by anyone yet. Bench
    caveat: the Python "chunk4096" column here (103–107 TF) carries a per-chunk output copy (~0.7 ms), the C++
    op in the PR writes in place (finding 75: 155–170 TF). Tile sweep: BN=128 halves throughput (42 TF, smem),
    BM=64 −5 %, 8 warps −17 %, swizzle group 4/8/16 within 3 %.

100. **CUTLASS tile-scheduler swizzle replaces chunking: `max_swizzle_size = 8` recovers 150–168 TF at every M,
    bit-identical (`swzbench`, standalone `_C_swz` = the PR's dispatch with `TileSchedulerArguments` exposed,
    two starts; `notes/data/swzbench.txt`).** The reviewer (gau-nernst) and the user's reading were right: the
    scheduler argument vLLM's `cutlass_gemm_caller` already accepts is the whole fix.

    | shape | M | stock (sw 1) | chunk4096 (PR) | sw2 | sw4 | **sw8 heuristic** | sw8 AlongM | sw8 AlongN |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 16384×2560 | 4096 | 165–170 | 151–161 | 154–164 | 153–162 | 149–155 | 148–154 | 145–156 |
    | | 8192 | 86–96 | 150–157 | 111 | 139–142 | **148–154** | 148–154 | 150–156 |
    | | 16384 | 52 | 151–156 | 93 | 142–144 | **152–156** | 151–155 | 152–156 |
    | | 32768 | 52 | 156 | 94 | 143 | **154** | 153 | 155 |
    | 10240×2560 | 8192 / 16384 / 32768 | 96 / 63 / 64 | 150 / 151 / 151 | 113 / 95 / 95 | 141 / 140 / 140 | **151 / 150 / 152** | 151 / 152 / 151 | 152 / 153 / 152 |
    | 5120×5120 | 8192 / 16384 / 32768 | 74 / 73 / 74 | 153 / 155 / 155 | 122 / 123 / 122 | 150 / 151 / 150 | **164 / 162 / 163** | 166 / 168 / 168 | 163 / 164 / 164 |

    Every configuration bit-identical to the single launch (swizzle only reorders CTAs). sw8 matches chunking on the
    skinny-N shapes and beats it by 6–8 % on 5120×5120 (chunking's per-launch tail cost); the raster order is
    within noise, heuristic is fine. Cost: at M=4096, where the weight already sits in L2, sw8 is 5–9 % below the
    stock order on 16384×2560 (155 vs 170) — so gate it exactly like the chunking was gated (`weight_bytes >
    l2CacheSize`, and M above ~1.5× the tile rows), or accept the small loss. Implication for PR #55180: drop
    the M loop, the scale re-layout, the chunk-size heuristic and the chunked-reference tests; keep the L2 gate
    and set `scheduler.max_swizzle_size = 8` on the kernel's `TileSchedulerArguments`. Also worth noting
    upstream: CUTLASS's default here is swizzle 1, and sw2 already halves the loss — the default is wrong for
    every part whose L2 is smaller than its weights.

101. **Nightly dev401 rebuild (`vllm-venv-fnmain2`) serves; first numbers within noise of dev352, agent turn
    possibly ~10 % faster (`fnmain2test2`, one start; `notes/data/fnmain2test2.txt`).** Build: clone + wheel +
    the dev401 overlay (17 files: #53899 offload worker port with the PLE gate merged onto #54882's mixed
    branch, the body SCALEINV rename, LMHEADQ, the two MTP-head patches, lmhead-scale loader). Two defects found
    on the first serve and fixed in the overlay: the SCALEINV rename predated the backup the overlay was
    diffed from (lost), and the merged gate returned None for a mixed checkpoint that lists the PLE under
    `exclude_modules` (ours) — now falls back to the config's `ple_embedding_dtype`. Results vs dev352:
    TTFT 8k 2.77 vs 2.71 s, 30k 10.68 vs 10.64 s (no spec); MTP n=3 + flag agent loop **1.92 s/turn vs
    2.10–2.15** (223 tok, acceptance 54.7 vs 57–60 %); hit intercept 0.592 s (unchanged, the block is
    unchanged: 1,600 with MTP, 1,568 without — the draft tokens enter the page math); hit +130 0.647 vs
    0.746 s, hit +1000 1.083 vs 0.978 s (mixed — the merged QSA kernel #54873 skips padded columns, which
    helps short chunks; the +1000 regression needs the three-start check before it is a finding). Kernel
    selection identical (`CutlassFp8BlockScaledMMKernel` everywhere incl. the MTP head), offload worker up.
    Not yet prod: the old venv stays; promote after a three-start agent loop and a `prefprof` on the new kernels.

102. **The pushed PR code verified standalone (`swz2bench`, `_C_swz2` = the branch's `.cu` + dispatch, two starts;
    `notes/data/swz2bench.txt`): 30/30 bit-identical to the stock kernel across 3 weights × 10 M values (64…16384),
    and 152–170 TF at every M ≥ 6144 on 16384×2560 (stock 137 → 52).** The sweep added the point that fixed the
    gate: M=5120 on 16384×2560 (12.5 MiB of A) still loses with the swizzle (153 vs 166), M=6144 (15 MiB) gains
    (153 vs 137), so the activation-slab threshold went from 12 to 14 MiB (commit bd84b180); 5120×5120 at M=4096
    (20 MiB) stays swizzled (117 → 160). The `m % 4 != 0` rows show the *preview* stock kernel's swap-AB misroute
    (15 TF) against the branch's main-based dispatch (#52775 fixed) — not a swizzle effect. PR #55180 state:
    three commits (rewrite, gate + balanced oracle, threshold), body v3, reviewer replies posted; done from our side.

103. **Tile-union QSA prototype works: 2.7× on the 8k prefill chunk, 1.7× on a 30k chunk, outputs equal to the
    stock kernel at bf16 rounding (`qsaunion2`, real selections from the dump, synthetic q/K/V at the model's
    geometry; `notes/data/qsaunion2.txt`; `tools/qsa_union_proto.py`).** Per program: R consecutive query rows
    share one gathered token set (the sorted union), a per-row membership mask over it, dot M = R·16.

    | dump | R / BN / warps | union columns per tile vs per row | speed vs stock | max diff |
    | --- | --- | --- | --- | --- |
    | 8k prefill chunk (3,813 rows, causal, mean sel 1,497) | 4 / 64 / 8 | 1,708 (1.14×) | **×2.69** (1.39 vs 3.76 µs/row) | 1e-4 |
    | 30k chunk (3,407 rows, 7.5k ctx) | 4 / 64 / 8 | 2,879 (1.41×) | **×1.74** (2.32 vs 4.04 µs/row) | 1e-4 |
    | 30k tail (283 rows, 12k ctx) | 4 / 64 / 8 | 3,858 (1.88×) | ×1.19 | 1e-4 |
    | 8k / 30k / tail | 2 / 64 / 8 | 1.08× / 1.17× / 1.36× | ×2.00 / 1.47 / 1.25 | 1e-4 |
    | 8k / 30k / tail | 8 / 32 / 8 | 1.20× / 1.74× / 2.60× | ×1.80 / 0.97 / 0.49 | 5e-4 |

    R=4, BN=64, 8 warps, one stage is the optimum (as finding 96 predicted; two stages do not fit the 99 KiB
    smem, 4 warps −10 %). The gain tracks the union width: where consecutive rows share most of their
    selection (early context, Jaccard ~0.9) the kernel gets the full small-M win; as the visible context grows
    the union widens (1.4× at 7.5k ctx, 1.9× at 12k) and the gain shrinks — so R should adapt (4 early, 2 late)
    or the union be built per 2 rows past ~8k of context. Two prototype defects fixed on the way: the
    membership search must run on a sorted key (padding at the end broke `searchsorted` and silently dropped
    tokens — the first run's outputs were wrong by the value magnitude), and each tile needs its own column
    bound (the first run looped every tile to the chunk-wide maximum). **Not yet counted: the union
    precompute** — 100–120 ms of torch per chunk per layer (sort/unique/searchsorted) would erase the gain
    across 12 attention layers; the real implementation needs a Triton merge of the (already sorted) per-row
    block lists at block granularity (512 ids per row, not 2,048 tokens), expanded ×4 inside the attention
    kernel. Expected TTFT effect with that in place: ~8 % at 8k (0.36 → ~0.14 s of 2.71), ~5 % at 30k.

104. **Union v2: precompute solved (0.7 ms per chunk, was 100 ms), kernel gains hold (`qsaunion3`,
    `notes/data/qsaunion3.txt`; `tools/qsa_union_v2.py`).** Union at block granularity (512 ids per row): one
    `torch.sort` over each tile's packed `(block_id*8 + row)` list plus a Triton kernel that flags first
    occurrences, prefix-sums the union position and scatters the union ids and the per-row membership; the
    attention kernel iterates union blocks (16 per step) and expands the four tokens itself. All outputs within
    1–2e-4 of the stock kernel.

    | chunk | R | union / row | precompute | kernel vs stock | total vs stock |
    | --- | --- | --- | --- | --- | --- |
    | 8k prefill, 3,813 rows | 4 | 1.12× | 0.74 ms | **×2.78** | ×2.43 |
    | | 2 | 1.06× | 0.71 ms | ×2.09 | ×1.89 |
    | 30k chunk, 3,407 rows at 7.5k ctx | 4 | 1.39× | 0.68 ms | **×1.80** | ×1.65 |
    | | 2 | 1.16× | 0.62 ms | ×1.52 | ×1.42 |
    | 30k tail, 283 rows at 12k ctx | 4 | 1.87× | 0.09 ms | ×1.23 | ×1.13 |
    | | 2 | 1.35× | 0.08 ms | ×1.30 | **×1.19** |

    A cost model fits all six points to within 5 %: kernel time ≈ tiles × union_count × c_R with c_4 = 12.8 ns and
    c_2 = 9.1 ns per (tile, union block) — R=2 is cheaper per unit (smaller M) but has more units. So the adaptive
    rule is: build both unions (1.4 ms) and pick the R with the smaller predicted time; R=4 wins up to ~8k of
    visible context, R=2 beyond. Precompute across 12 attention layers ≈ 9–17 ms per 8k prefill against ~0.2 s
    saved. Next: integration behind `VLLM_QSA_UNION=1` in the nightly venv (single-request prefill chunks first,
    per-request tiles later), then the server-level TTFT.

105. **Union v2 integrated, server level: −2 % TTFT at 8k, 0 at 30k (`qsaunion5`, nightly venv, batch 4096, two
    interleaved starts; `notes/data/qsaunion5.txt`).** 8k: 2.67 / 2.67 s on vs 2.72 / 2.73 off; 30k: 10.57 / 10.59 vs
    10.54 / 10.58. Standalone the same code was 1.51× on the 8k chunk, 1.17× at 30k and 0.86× on the 283-row tail
    (`qsaunion5` test lines): the per-call fixed cost — two unions per call with 4,096-wide sorts (the three tail
    entries pushed R·E past 2,048), three device→host reads — ate most of the kernel's gain. The 2026-09-04 review
    named the fixes; v3 (exact-width sort, one union by context, gate 1,024 rows) and v4 (block-only union at
    2,048/1,024, tail as a separate 16-column pass, ratio/metadata threaded, no device reads, request validation,
    asserting test) are queued behind this run, then the v5 standalone (union from block ids, R-bit membership,
    pre-resolved physical pages, R=2 tile sweep).

106. **Union v3 integrated, server level: −4 % TTFT at 8k, −1 % at 30k, two starts (`qsaunion6`,
    `notes/data/qsaunion6.txt`).** 8k: 2.62 / 2.61 s on vs 2.72 / 2.73 off; 30k: 10.44 / 10.42 vs 10.55 / 10.56.
    Standalone: 1.97× on the 8k chunk, 1.40× at 30k, 1.00 on the tail (gated off below 1,024 rows). v3 = exact-width
    sort, one union chosen by the last row's max index, request read in-kernel, 1,024-row gate. Doubling of the
    server gain vs v2 (finding 105) came from the halved precompute; the remaining gap to the finding-104 target
    (~8 %) is the 4,096-wide build (three tail entries per row) and the R choice, which v4 removes.

107. **Union v5 standalone sweep (`qsaunion8`, `tools/qsa_union_v3.py`, `notes/data/qsaunion8.txt`): R=2 at BN=32
    is the best tile so far, R=4 collapses in this form.** v5 = union built from the 512 block ids directly (no
    expansion), R-bit membership mask per union block (`atomic_or`), physical pages pre-resolved in the
    precompute, causal tail from the query positions; sweep R ∈ {4, 2} × BN ∈ {32, 64, 128} × warps ∈ {4, 8}
    (BN=128 needs 147–164 KiB of smem, over the 99 KiB cap at either R). Kernel / total (with precompute) vs stock:

    | chunk | R=2 BN=32 w4 | R=2 BN=64 w4 | R=4 BN=64 w4 | R=4 BN=64 w8 | R=4 BN=32 | precompute |
    | --- | --- | --- | --- | --- | --- | --- |
    | 8k, 3,813 rows | **2.36 / 1.88** | 2.02 / 1.66 | 1.51 / 1.30 | 1.27 / 1.12 | 0.06–0.17 | 1.55 ms |
    | 30k chunk, 3,407 rows at 7.5k | **1.75 / 1.49** | 1.51 / 1.31 | 1.03 / 0.93 | 0.94 / 0.86 | 0.04–0.11 | 1.42 ms |
    | 30k tail, 283 rows at 12k | **1.44 / 1.06** | 1.26 / 0.96 | 0.72 / 0.61 | 0.66 / 0.57 | 0.03–0.09 | 0.30 ms |

    Against the v2 kernel (finding 104: R=4 2.78×, R=2 2.09× at 8k) the v5 form gains at R=2 with the narrower
    tile (2.09 → 2.36) and loses badly at R=4 (2.78 → 1.51; the BN=32 arm is 60× slower than stock — the
    signature of register spilling once the 64×256 fp32 accumulator shares the file with the mask expansion and
    the pre-resolved int64 addresses). The v4.3 kernel (finding 108) has the same tail pass but the int8
    membership matrix and the in-kernel page lookup, and keeps R=4 ahead of R=2 — so the loss is in the bitmask
    or the pre-resolved addressing, not in the tail pass. Precompute doubled vs v2 (0.7 → 1.5 ms): the page
    pre-resolution and the tail gather are extra torch ops. Bisect queued (v6: bitmask vs matrix × pre-resolved
    vs table, at R=4 BN=64 and R=2 BN=32).

108. **Union v4.3 integrated: asserting test green on three dumps, standalone 1.82× / 1.36× / 1.07×
    (`qsaunion12` test lines; `tools/qsa_union_test.py`).** v4 as reviewed (block-only union at exact 2,048/1,024,
    causal tail as a separate 16-column pass in the same online softmax, ratio/top-k/context/request count from
    the owner's metadata, no device reads, stock request validation), plus three fixes the new test forced:
    the tile's block-table row comes from any valid row of the tile (the first row may be padding); rows with an
    invalid request id are excluded from the softmax (a zero query still attends uniformly otherwise) and are
    written as zeros like the stock kernel. The test uses a peaked softmax (|q| ≈ 2, scores ~N(0, 4)) so a
    dropped or leaked token moves the output by ~0.1 against ~1e-3 of bf16 noise; a negative control (one block
    swapped per row) asserts that power; every tail length 0..3, both R, a permuted physical-page table with
    two decoy requests, the +1 count column of newer main, invalid-request rows, and the CPU-only eligibility.
    Whole-path timing (split + build + kernel) vs stock at R=4 / R=2: 8k 1.82× / 1.55×, 30k chunk 1.36× / 1.23×,
    tail 0.91× / 1.07× — below v2's 2.43× and v3's 1.97× at 8k although the sort is narrower, so the torch glue of
    the split and build (a dozen small ops) now costs more than the kernel gains; next the components are timed
    separately and the split goes away (lever 1: build from `block_indices`). Server A/B running.

110. **Union v4.3 integrated, server level: −4 % TTFT at 8k, −1 % at 30k, two starts (`qsaunion12`,
    `notes/data/qsaunion12.txt`) — identical to v3 (finding 106).** 8k: 2.65 / 2.63 s on vs 2.73 / 2.74 off; 30k:
    10.53 / 10.48 vs 10.55 / 10.58. The review rework (block-only union, separate tail pass, metadata-threaded
    parameters, no device reads, request validation, asserting test) is thus correctness- and hygiene-neutral on speed:
    the standalone whole-path went 1.97× → 1.82× at 8k (finding 108) and the server did not move, so the server number
    is set by something the standalone does not contain — the per-layer torch glue around the call (split + build
    ≈ 15 small launches × 12 layers per chunk) and the stock path's own share of the chunk. Target remains ~8 %
    (finding 103); the levers are now (1) build from `block_indices` on the indexer (no split), (2) fuse the build's
    torch ops into the Triton kernel, (3) the v6 bisect result for the R=2 BN=32 tile.

111. **v6 bisect (`qsaunion13`, `tools/qsa_union_v6.py`, `notes/data/qsaunion13.txt`): the R-bit membership mask is
    the R=4 regression; pre-resolved pages are a consistent +5–8 %; R=2 at BN=32 with 4 warps is the best tile at
    both contexts.** Kernel vs stock, 2×2 of membership form × addressing:

    | tile | 8k: bits+phys / bits+table / **matrix+phys** / matrix+table | 30k chunk (7.5k ctx): same order |
    | --- | --- | --- |
    | R=4 BN=64 w8 | 1.23 / 1.22 / **2.65** / 2.50 | 0.93 / 0.94 / **1.79** / 1.70 |
    | R=4 BN=64 w4 | 1.53 / 1.48 / **2.46** / 2.14 | 1.06 / 1.01 / **1.63** / 1.43 |
    | R=4 BN=32 w8 | 0.17 / 0.17 / **1.96** / 1.81 | 0.11 / 0.11 / **1.29** / 1.20 |
    | **R=2 BN=32 w4** | 2.36 / 2.18 / **2.71** / 2.53 | 1.75 / 1.63 / **1.94** / 1.86 |
    | R=2 BN=64 w4 | 2.11 / 1.91 / 2.14 / 1.92 | 1.58 / 1.41 / 1.57 / 1.42 |

    The `(um[None, :] & (1 << r)[:, None]) != 0` expansion is what spills at M=64 (a [M, BNB] int32 temporary and its
    broadcast to [M, BN] on top of the 64×256 accumulator — 60× slower at BN=32); the int8 [M, BNB] load has no such
    temporary. Lever 2 (bitmask) is therefore rejected; lever 3 (pre-resolved physical token bases, one [BNB] load
    instead of the page-table gather in the loop) is kept. With matrix + pre-resolved, R=2 BN=32 (4 warps) beats the
    finding-104 optimum R=4 BN=64 at 8k (2.71 vs 2.65) and clearly at 7.5k ctx (1.94 vs 1.79), and the R=2 union is
    the narrower one (1.06× / 1.16× of a row vs 1.12× / 1.39×) — so one fixed tile replaces the adaptive R choice,
    and the build shrinks to N = 1,024. v7 = the v4.3 kernel with those two changes.

113. **Union v7 (R=2, BN=32, 4 warps, pre-resolved pages) at the server: +1 % / +1 % — the standalone win did not
    carry (`qsaunion14`, `notes/data/qsaunion14.txt`).** 8k: 2.78 / 2.74 s on vs 2.73 / 2.73 off; 30k: 10.79 / 10.59 vs
    10.55 / 10.59. Standalone the same code was the best so far (whole path 1.89× / 1.51× / 1.25×, finding 108 → this
    run's test lines: split 1.2 ms + build 0.8 ms + kernel 5.6 ms at 8k). The standalone is L2-resident: three to eight
    1600-token pages of K/V (1.6 MiB each) fit in the 24 MiB L2, so its gathers are free and the smaller M of R=2
    wins on the dot; in the server the K/V pages come from DRAM and the union's point — one gather shared by R rows —
    is worth more at R=4. The asserting test now also times a cold cache (2,048 pages, random table); the v8 run
    measures R=2, R=4 and off at the server. Lesson for the standalone: size the cache past the L2 before ranking tiles.

114. **v7's and v8's server arms never ran the union kernel: the layout guard fell back to stock.** `forward_qsa`
    hands the kernel `kv_cache.transpose(1, 2).split(head_size, dim=-1)` — K and V are [blocks, PAGE, kv, D] views of a
    wider tensor, so block stride ≠ PAGE × token stride, and v7's `qsa_union_layout_ok` (written for the standalone's
    contiguous cache) returned False on every call. Finding 113's "+1 %" is therefore stock + noise, not a tile
    ranking, and the L2 explanation there is withdrawn; `qsaunion15` (v8 arms) was stopped for the same reason.
    v9 stores page × PAGE + offset per union block and decomposes it in the attention loop (page × stride_block +
    offset × stride_token), which is layout-free and costs nothing (standalone raw path 2.26× at 8k, identical to
    v8), and logs the path it takes once per process (`QSAUNION path: raw …` / `split …` / `stock fallback …`) so a
    silent fallback can never again pass for a measurement. The v8 test lines (`notes/data/qsaunion15.txt`) stand as
    the standalone result for lever 1: raw build 0.8 ms replaces split + build 2.0 ms; whole path R=2 2.24× / 1.71× /
    1.41× (8k / 30k chunk / tail), R=4 1.92× / 1.33× / 0.90×. `qsaunion16` = v9 at the server, R=2 / R=4 / off, two
    starts each, with the path line per arm.

115. **Union v9 at the server: −4.7 % TTFT at 8k, −3.7 % at 30k with R=2; R=4 is −3 % / +0.5 %; two starts each
    (`qsaunion16`, `notes/data/qsaunion16.txt`).** Every union arm's log carries `QSAUNION path: raw (indexer
    selection)`, so this is the kernel, not a fallback.

    | arm | 8k (7,503 tok) | 30k (29,263 tok) |
    | --- | --- | --- |
    | **R=2, BN=32, 4 warps** | **2.60 / 2.59 s** | **10.17 / 10.18 s** |
    | R=4, BN=32, 4 warps | 2.64 / 2.65 s | 10.59 / 10.64 s |
    | off | 2.72 / 2.74 s | 10.57 / 10.57 s |

    Per-start spread ≤ 0.01 s at 8k and ≤ 0.05 s at 30k, so the ordering is not noise. R=2 wins at both contexts,
    as the standalone said (finding 111) — and at 30k R=4 is a wash because its union widens to 1.4× a row while the
    tile count halves; the cost model of finding 104 predicted exactly that crossover, at ~8k. Progress on the
    ~8 % target (finding 103): v3/v4.3 −4 % / −1 % → v9 −4.7 % / −3.7 %; the 30k gain is new and comes from lever 1
    (no split) plus the R=2 tile. What is left: the build is still ~0.8 ms of torch launches per call (sort + six
    small ops × 12 layers × chunks), the 1,024-row gate leaves the 283-row tail chunks on the stock kernel (1.41×
    standalone at R=2 — the dispatch table of the review's lever 4), and the kernel itself sits at 2.5× while the
    dot at M=32 is far from the tensor-core roofline. Default tile is now R=2 (`VLLM_QSA_UNION_R`).

117. **Upstream branch head 8c09f0c5 at the server (`tuval`, `notes/data/tuval-8c09f0c5.txt`): −2.6 % TTFT at 8k,
    −2 % at 30k, concurrent pairs −2.5 % / −3 %, warm turns unchanged; two starts; path verified by the dispatch,
    warmup and prefix-caching lines per arm.** First valid server number for the branch (the first two runs on it
    measured stock: a layout guard, then the int64 positions contract — findings 114/116's lesson, again).

    | | union (auto) | off | Δ |
    | --- | --- | --- | --- |
    | 7,503 tok | 2.59 / 2.59 s | 2.67 / 2.65 s | −2.6 % |
    | 29,263 tok | 10.15 / 10.14 s | 10.40 / 10.30 s | −2.1 % |
    | 8k + 8k concurrent, pair wall | 5.21 / 5.19 s | 5.37 / 5.32 s | −2.7 % |
    | 30k + 8k concurrent, pair wall | 12.57 / 12.74 s | 13.07 / 12.84 s | −2.3 % |
    | 8-turn agent loop, MTP 3 + prefix cache, s/turn | 1.68 | 1.63 | +3 % (one start; turns 1.2–1.6 s both, noise) |

    The union arm reproduces v9 exactly (2.59 / 10.15 vs 2.59–2.60 / 10.17–10.18): the review's preprocessing did
    not cost anything. The gain shrank because the **reference moved**: this run's off arm runs current main's QSA
    files (#54873 and later) instead of the dev401 nightly's, and those are 2–3 % faster on their own (2.67 vs 2.73
    at 8k, 10.30–10.40 vs 10.55–10.59 at 30k). So the honest PR claim on today's main is −2.6 % / −2 %, not
    −4.7 % / −3.7 %. Concurrent batches with mixed requests keep the same ratio (the per-request tile map works
    at no cost), warm agent turns are unaffected (below the 1,024-row gate; acceptance identical). Next: the same
    A/B on head 07a6d2a3 (expansion skipped for non-reused layers, tail-only zeroing, shared layout and workspace,
    int64 tails) — chain 2, queued.

118. **Branch head 30f3446d (deferred items) at the server (`tuval2`, `notes/data/tuval2-30f3446d.txt`): −2.8 % TTFT
    at 8k, −1.7 % at 30k, pairs −2.2 % / −1.6 %, warm turns unchanged; two starts; path lines per arm.** Same harness
    as finding 117, one hour later, same off reference.

    | | union (auto) | off | Δ | finding 117 (union) |
    | --- | --- | --- | --- | --- |
    | 7,503 tok | 2.58 / 2.57 s | 2.65 / 2.65 s | −2.8 % | 2.59 / 2.59 |
    | 29,263 tok | 10.13 / 10.09 s | 10.28 / 10.28 s | −1.7 % | 10.15 / 10.14 |
    | 8k + 8k pair | 5.20 / 5.19 s | 5.31 / 5.31 s | −2.2 % | 5.21 / 5.19 |
    | 30k + 8k pair | 12.61 / 12.55 s | 12.80 / 12.79 s | −1.6 % | 12.57 / 12.74 |
    | agent loop, s/turn | 1.69 | 1.72 | noise | 1.68 |

    The deferred items — expansion skipped for layers the proposer never flags for reuse, tail-only output zeroing,
    one row → tile layout per forward, one selection workspace per device, int64 tails — are worth 10–50 ms per
    request (0.4 %), inside the run-to-run band but never negative, and the warm loop with MTP 3 confirms the
    reuse flag: the drafter's layers kept their expansion, acceptance and turn times unchanged. Two defects the
    kernel test cannot see surfaced only at the serve start (a `register_buffer` name clash, a `head_size` on the
    wrong object); both are layer-construction/forward issues — the pytest never builds the owner layer.
    Verdict for the PR body: **−2.8 % / −1.7 % TTFT on today's main, single-request and mixed-request batches
    alike, no effect on decode or warm turns**; the −4.7 % / −3.7 % of finding 115 was against the older nightly's
    stock kernel and is not the claim.

119. **Fragmented prefill batches (`tufrag`, `notes/data/tufrag.txt`): the union is neutral on short-context
    multi-request batches — no loss where it is eligible, no gain either; the gate is not costing anything.** Same
    branch, `--max-num-seqs 128`, batch 4096, N salted prompts fired together, pair wall, medians of 3, one start each.

    | batch | tokens | union | off | eligible? |
    | --- | --- | --- | --- | --- |
    | 1 × 4k | 4,106 | 1.47 s | 1.49 s | yes |
    | 4 × 1k | 4,180 | 1.54 s | 1.53 s | yes |
    | 16 × 260 | 4,214 | 1.73 s | 1.71 s | yes (260 rows/request) |
    | 64 × 94 | 6,006 | 3.03 s | 3.01 s | yes (94 ≥ 64) |
    | 128 × 61 | 7,826 | 4.46 s | 4.41 s | no (61 < 64 → stock) |

    All within ±1.5 %, one start, so noise. Reading: at these context lengths every row's selection is the whole
    (short) context, so the union saves gathers but the split-K kernel is already cheap there — the union's gain is a
    long-context effect (finding 111's cost model: it scales with the selection width, which is ≤ context/CR here).
    The per-request gate of 64 rows therefore neither protects nor costs anything measurable on this box; it stays as
    the conservative default (a fragmented batch's tiles share little, and the build is fixed cost) and the override's
    fifth field lets other parts move it. Evidence for the PR body's "mixed-request batches the same": yes at 8k+8k
    (finding 117/118), neutral below.

120. **The stock split-K QSA kernel's config table, retuned on GB10 (`tune`, `tools/qsa_splitk_tune.py`,
    `notes/data/tune-splitk-gb10.txt`): the GB300 table over-splits on 48 SMs; batched decode/verify gains 1.1–1.6×,
    prefill 1.05× (1.25× on tail chunks).** One run per cell, medians of 5×5, real prefill dumps + synthetic uniform
    decode batches; outputs within 1e-4 of the stock config. gau-nernst's suggestion (RFC #55394, 06:44).

    | shape (rows × requests) | base programs | stock (BN, splits, warps) | best | gain |
    | --- | --- | --- | --- | --- |
    | 1 × 1, 8k ctx | 2 | 32, 64, 4 | 32, 16, 1 | 1.02× |
    | 4 × 1 | 8 | 32, 64, 4 | 32, 8, 1 | 1.07× |
    | 4 × 4 | 8 | 32, 64, 4 | 16, 16, 4 (32, 8, 1 within 1 %) | **1.61×** |
    | 16 × 4 | 32 | 32, 16, 1 | 64, 1, 2 | **1.34×** |
    | 32 × 8, 32k | 64 | 32, 8, 1 | 64, 1, 4 | 1.12× |
    | 64 × 16 | 128 | 32, 4, 1 | 64, 1, 2 | 1.11× |
    | 128 × 32 | 256 | 32, 8, 1 | 64, 1, 2 | **1.21×** |
    | 512 × 128 | 1024 | 64, 1, 2 | same | 1.01× |
    | prefill 3,813 / 3,407 rows | > 2048 (prefill) | 32, 1, 1 | 32, 1, 8 | 1.05× |
    | prefill tail 283 rows | 566 (prefill) | 64, 1, 2 | 16, 1, 4 | **1.25×** |

    Reading: the GB300 table's 64-way split at ≤ 24 base programs makes 512 tiny programs plus a merge on a 48-SM
    part; 8–16 splits are enough here. From 32 base programs up, no split at all with BN = 64 wins, and the prefill
    branch wants 8 warps at BN = 32. A GB10 table therefore: bp ≤ 24 → (32, 8, 1); ≤ 256 → (64, 1, 2) (4 warps at
    64); ≤ 1024 → prefill (16, 1, 4) / decode (64, 1, 2); > 2048 prefill → (32, 1, 8). **Caveats before a PR:** one
    run per cell; the decode cells' K/V (6–20 pages) sit in the 24 MiB L2, unlike a real decode step; bp 512 and
    decode at > 1024 rows unmeasured; the merge kernel's cost under CUDA graphs differs from eager timing. Needs
    2–3 repeats, the missing shapes, and a server A/B (decode c=1/4/16 tok/s + TTFT) before the numbers are claimed.
    A device-keyed table (CC 12.x) in `_select_config` plus this sweep under `benchmarks/kernels/` is the short PR
    he offered to take.

    Side observation that matters for the union question: the same 8k dump runs the stock kernel at 9.4 ms here
    (3-page, L2-resident cache) vs 14.6 ms in `test_qsa_tile_union.py` (18 pages, DRAM) — the union's standalone
    2.24× was measured against the DRAM-bound stock, and an 8k prompt's K/V (~25 MiB) straddles the L2 in the server.
    The in-situ profile (`tuprof`) settles what the union kernel and the stock kernel actually cost per call there.

121. **In situ, the tile-union kernel is 1.45× the split-K kernel, not the replay's 2.7×; the integration loses
    nothing (`tuprof`, `notes/data/tuprof.txt`; traces in `/opt/llm/runners/results/traces/`;
    `tools/prof_summary.py`, `tools/prof_border.py`).** One 7,507-token request per arm under the torch profiler,
    same branch, union auto vs 0, batch 4096 (two chunks: 4,096 + 3,411 rows), 12 QSA layers = 24 calls:

    | | union on | off |
    | --- | --- | --- |
    | attention kernel, 24 calls | 182.3 ms (7.6 ms/call; 6.4–7.1 first chunk, 8.3–9.2 second) | 265.1 ms (11.0 ms/call) |
    | glue kernels between top-k and attention | 10.4 ms (sort 3.8, layout ops 2.0, pack 1.35, build 1.23; + KV write 2.0 in both) | 6.3 ms (expand 4.2) |
    | GPU idle inside that window | 3.6 ms (1.9 %, 0.15 ms/call) | 0.2 ms |
    | gap before the attention kernel | 3 µs | 3 µs |
    | QSA-related kernels, share of GPU time | 7.4 % of 2.55 s | 10.5 % of 2.59 s |
    | TTFT of the profiled request | 2.64 s | 2.69 s |

    Arithmetic: −83 ms of attention kernel, +4 ms of glue, +3 ms of idle = −76 ms ≈ 2.9 % — exactly the e2e gain of
    findings 117/118. **Where the replay's 2.7× went — corrected by the three-way run (`threeway`,
    `notes/data/threeway.txt`):** not the L2. With the cache spread over 64 pages the stock kernel still takes 9.4 ms
    on the 8k dump, the same as with 3 pages; the 14.6 ms baseline of every union replay up to finding 115 was the
    **pre-#54873 split-K kernel** (the dev401 nightly is 8340fe1bb, built 12:27 UTC on 09-04; #54873 "Improve QSA
    sparse GQA for prefill and short-ctx decode" merged 13:08 UTC; the venv carried the old kernel until the branch
    overlay that evening — its `ops/qsa.py.orig-dev401` has no packed count column). gau-nernst's kernel is 1.55×
    faster than the one we measured against. Against it: replay 9.4 → 6.3 ms (1.50×), 7.5k-context chunk 11.4 → 8.0 ms
    (1.42×), in situ 11.0 → 7.6 ms (1.45×) — consistent. The GB10-tuned config of finding 120 adds 1.03–1.05× on
    top of stock in the same replay. The host side is not a factor: 0.15 ms idle per call, 3 µs launch gap.
    Consequence for the RFC: the union's real edge on this box is ~1.45× on a kernel that is 10 % of prefill — the
    ~3 % end to end is the ceiling of this design here, not an integration loss. A three-way (stock / GB10-tuned /
    union) under both cache states follows (`threeway`), then the server A/B of the tuned table (`tuchoice`).

123. **Tuning vs union at the server (`tuchoice`, `notes/data/tuchoice.txt`): the GB10 split-K table gives −1.5 % / −1.3 %
    TTFT and +8 % MTP single-stream decode; the union on top gives a further −1.9 % / −1.4 % TTFT; they stack.** One start
    per arm, `--max-num-seqs 16`, 4 GiB KV, batch 4096. Cold arms: stock table, GB10 table, GB10 table + tile-union.
    Warm arms (MTP n=3, prefix cache on): stock table, GB10 table.

    | | stock table | GB10 table | GB10 table + union |
    | --- | --- | --- | --- |
    | TTFT 7,503 tok | 2.66 s | 2.62 s (−1.5 %) | 2.57 s (−3.4 %) |
    | TTFT 29,263 tok | 10.38 s | 10.24 s (−1.3 %) | 10.10 s (−2.7 %) |
    | 8k + 8k pair | 5.34 s | 5.29 s | 5.17 s (−3.2 %) |
    | 30k + 8k pair | 13.02 s | 12.70 s (−2.5 %) | 12.56 s (−3.5 %) |
    | decode no-spec, c = 1 / 4 / 16 | 24.7 / 65.1 / 146.0 | 24.2 / 67.7 / 147.1 | 22.7 / 68.9 / 149.0 |
    | decode MTP 3, c = 1 | 39.2 | **42.4 (+8 %)** | — |
    | decode MTP 3, c = 4 | 98.0 then 42.1 | 94.9 then 42.8 | — |
    | decode MTP 3, c = 16 | 95.0 / 95.1 | 98.1 then 78.2 | — |
    | agent loop, s/turn | 1.72 | 1.66 | — |

    Reading: on prefill the union is worth twice the table (each is a 10 %-of-prefill kernel moved 1.05× vs 1.45×).
    On decode only the table acts. **The MTP c=1 "+8 %" (39.2 → 42.4) is NOT attributable to the table's kernel time:**
    at the verify shape (4 rows per request) the sweep's saving is 2.7 µs per call, ~35 µs per 25 ms step across 12
    layers (0.1 %), and the drafter's 1-row steps are unchanged. Byte-identical prompts, but a different BLOCK_N /
    split count changes the summation order, near-tie tokens flip, the text diverges and the accepted draft length
    with it — the channel of our #54521 measurement (32.8 vs 64.2 ms/tok on acceptance alone) — on top of the 6.9 %
    decode noise band. Unattributed until acceptance is read per arm (`accept.py`) and the cell repeated. No-spec
    decode at c=4 gains 4 % (67.7 vs 65.1, tight reps; the 4×4 shape where the sweep found 1.61× at kernel level —
    plausible but one start); c=1 and c=16 are within noise (the union arm's 22.7 at c=1 included — the union never
    runs on decode rows). **Anomaly, both warm arms:** the MTP c=4 cell is bimodal — 95–98 tok/s on the first repetition, ~42 on the
    second, with prefix caching on; c=16 on the GB10 arm shows the same drop (98 → 78). Not a table effect. Same shape as
    DJLougen's acceptance collapse on batch geometry; the probe does not log acceptance — a dedicated check (acceptance
    per repetition, cache on/off) is queued in the TODO before that cell is quoted anywhere.
    Decision input: the table is the cheap PR (a device-keyed entry; −1.5 % TTFT, +4 % no-spec decode at c=4, MTP
    effect unproven); the union adds −2 % TTFT on top for ~900 lines. Both are honest against #54873's kernel.

124. **Boundary shapes where #54873 has the most headroom (`tubound`, `notes/data/tubound.txt`): the union never
    loses, and below ~2k context it does not win either.** Union auto vs 0, one start each, three repetitions, pair wall.

    | batch | tokens | union | off | note |
    | --- | --- | --- | --- | --- |
    | 1 × 1,011 | 1,011 | 0.48 s | 0.48 s | below the 1,024-row gate → stock in both arms |
    | 2 × 501 | 1,002 | 0.63 s | 0.63 s | below the gate → stock |
    | 1 × 1,521 | 1,521 | 0.62 s | 0.62 s | eligible; neutral |
    | 1 × 2,031 | 2,031 | 0.77 s | 0.78 s | eligible; neutral |
    | 1 × 4,106 | 4,106 | 1.45 s | 1.49 s | eligible; −2.7 % |
    | 16 × 59 | 950 | 0.69 s | 0.70 s | below both gates → stock |

    The 1×1024 cell missed the gate by 13 tokens (the filler unit is 34 tokens); an exact-boundary cell (1×1,045) is
    the one shape not covered — but 1,521 and 2,031 rows, where every row's selection is still short of the sparse
    budget and #54873's `valid_count` pruning is strongest, come out exactly even, so the union's sort/build at the
    fixed 1,024-key width does not lose against the pruned split-K there. The gain starts where the selection
    saturates (4k: −2.7 %; 8k: −2.8 %). No gate change needed; the `effective_block_topk` sort-width idea stays
    unimplemented (it would only matter below 1k context, which the gate excludes).

125. **Swizzle N/K sweep for PR #55180 (`swzshapes`, `notes/data/swzshapes.txt`): the swizzled order is flat at
    150–174 TF at every M; the default order is the erratic one. gau-nernst's simpler rule — swizzle iff the weight
    exceeds the L2 — is right on both sides, and the activation-slab term is dropped.** Ten shapes × M 2048–16384,
    stock (default order) → max_swizzle_size 8, TF, bit-identical on every cell:

    | N×K | weight | M=2048 | M=4096 | M=6144 | M=8192 | M=12288 | M=16384 |
    |---|---|---|---|---|---|---|---|
    | 16384x2560 | 40 MiB | 153→172 (×1.12) | 167→153 (×0.92) | 146→153 (×1.04) | 93→152 (×1.63) | 54→156 (×2.92) | 53→156 (×2.93) |
    | 12288x2560 | 30 MiB | 153→160 (×1.04) | 160→154 (×0.96) | 107→152 (×1.43) | 93→153 (×1.65) | 53→153 (×2.87) | 53→153 (×2.91) |
    | 10240x2560 | 25 MiB | 147→157 (×1.07) | 155→152 (×0.98) | 145→152 (×1.05) | 90→154 (×1.72) | 70→152 (×2.18) | 70→154 (×2.22) |
    | 2560x6144 | 15 MiB (fits L2) | 178→163 (×0.92) | 161→160 (×0.99) | 172→165 (×0.96) | 171→164 (×0.96) | 168→166 (×0.99) | 172→168 (×0.97) |
    | 5120x5120 | 25 MiB | 164→162 (×0.99) | 123→166 (×1.34) | 76→168 (×2.21) | 86→164 (×1.90) | 82→168 (×2.06) | 77→169 (×2.20) |
    | 7168x5120 | 35 MiB | 166→164 (×0.99) | 123→164 (×1.33) | 53→172 (×3.26) | 58→170 (×2.92) | 57→170 (×2.96) | 58→169 (×2.90) |
    | 4096x4096 | 16 MiB (fits L2) | 162→163 (×1.01) | 158→161 (×1.01) | 163→163 (×1.00) | 163→164 (×1.00) | 165→164 (×0.99) | 160→164 (×1.03) |
    | 8192x8192 | 64 MiB | 158→159 (×1.01) | 72→168 (×2.32) | 54→174 (×3.21) | 53→171 (×3.21) | 54→172 (×3.22) | 54→171 (×3.19) |
    | 14336x4096 | 56 MiB | 161→165 (×1.03) | 153→164 (×1.07) | 62→164 (×2.66) | 56→166 (×2.99) | 50→166 (×3.31) | 52→167 (×3.24) |
    | 32768x4096 | 128 MiB | 165→167 (×1.01) | 160→162 (×1.02) | 65→165 (×2.54) | 54→166 (×3.06) | 52→167 (×3.23) | 53→168 (×3.20) |
    
    bit-identical everywhere: True

    Reading: (1) where the weight exceeds the 24 MiB L2, swizzle 8 is equal or up to 3.3× faster at every M except a
    narrow band at M = 4096 on the 2560-wide weights (0.92–0.98) — the band my activation-slab gate was built around;
    that gate also kept the default order at M = 2048 where swizzle wins 12 % on 16384×2560, so it was net wrong.
    (2) Where the weight fits the L2 (2560×6144 at 15 MiB, 4096×4096 at 16 MiB) the swizzle is neutral to −8 %,
    which the `weight > L2` condition excludes. (3) For sm120 parts (96–128 MiB L2) no weight here exceeds the L2,
    so the kernel launch is unchanged there by construction. PR updated (gate = weight > L2 only; tests folded into
    the existing blockwise test with two prefill-sized cases).


126. **MTP decode cells with acceptance per repetition (`acceptcell`, `notes/data/acceptcell.txt`): the split-K table
    has no MTP decode effect, every tok/s difference is acceptance, and the c=4 collapse is a real drafter failure that
    reproduces on two of three arms, with the prefix cache on and off.** Stock table / GB10 table (both prefix cache on)
    / stock with cache off; MTP n=3, trailing-block drop disabled, 4 GiB KV, 128 new tokens, 543-token prompt, three
    repetitions at c=1 and c=4, two at c=16; acceptance from `/metrics` deltas around each cell.

    | cell | stock | GB10 table | stock, cache off |
    |---|---|---|---|
    | c=1 tok/s (acceptance) | 37.1 (60 %), 44.0 (75 %), 42.9 (74 %) | 39.6 (65 %), 43.4 (73 %), 39.2 (61 %) | 40.3 (75 %), 40.9 (69 %), 40.9 (68 %) |
    | c=1 mean | 41.3 | 40.7 | 40.7 |
    | c=4 tok/s (acceptance) | 94.2 (71 %), 93.9 (66 %), 94.5 (70 %) | 94.6 (68 %), 96.9 (72 %), **42.8 (9 %)** | 89.5 (66 %), **42.3 (9 %)**, 93.9 (70 %) |
    | c=16 tok/s (acceptance) | 98.5 (67 %), 101.3 (68 %) | 97.3 (68 %), 97.3 (68 %) | 98.6 (68 %), 98.5 (69 %) |

    Reading: (1) c=1 spans 37–44 tok/s *within one arm* and the order is the acceptance order (60 → 75 % is
    37.1 → 44.0); the arm means are within 0.6 tok/s. Finding 123's "+8 % MTP c=1 from the table" was this channel,
    as suspected there; withdrawn. (2) The healthy c=4 cells are 94–97 on every arm, c=16 is 97–101 — and c=16 runs
    five wide: the 4 GiB KV budget holds five requests (18.8 % usage each, `Running: 5, Waiting: 11` in the log), so
    those cells are queue-limited and not a 16-stream number. (3) **The collapse:** one c=4 cell per arm on two arms
    drops to 42 tok/s with 9.1 % draft acceptance (per-position 0.09 / 0.06 / 0.05), then the next cell is normal
    again. Not the table (stock arm 3/3 healthy, GB10 arm rep 2), not the prefix cache (cache-off arm rep 1), not
    preemption or any logged event (nothing in the server log for the window; KV at 56–75 %). The two collapsed cells
    have the same structure to within one step — 400 / 401 draft steps, 109 / 110 accepted for 512 tokens — which is
    what three requests with a dead drafter (≈1 token per step) plus one healthy request (≈42 steps) produce, and
    the log shows `Running: 3` mid-cell where four were started. The target's text is unaffected as far as the probe
    shows (the same 60-character opening as the healthy cells), so this is the drafter's per-request state, not the
    model: the MTP layer carries a PLE short-conv state with a spec-step rollback and a reused step-0 QSA selection
    (`compact_topk_indices`), either of which can go stale for a request without touching the target. Same family as
    DJLougen's batch-geometry acceptance collapse. Per-request attribution (latency, full text, preemption counter)
    over 10 repetitions at c = 1, 2, 3, 4, 5, 8 on three configs is running as `acceptcell2` → next finding. Until
    then, **no MTP c ≥ 4 number is quotable** and the union/table decision rests on TTFT and the no-spec decode cells.
