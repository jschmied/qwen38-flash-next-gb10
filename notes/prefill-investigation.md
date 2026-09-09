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
    over 10 repetitions at c = 1, 2, 3, 4, 5, 8 is in **finding 127 (determinism-investigation.md): it is the target output that is corrupted, for all but one of the requests prefilled in the same step; MTP-specific, production build affected.** Until
    then, **no MTP c ≥ 4 number is quotable** and the union/table decision rests on TTFT and the no-spec decode cells.

132. **GB10 tuning profiles, measured (`gb10tune`, `gb10tune2`; `notes/data/gb10tune*.txt`): the split-K table is at the
    kernel-level optimum on every production shape and has NO server-level effect at three starts; the pre-indexer and the
    scoring kernel are microseconds; the cooperative top-k cannot launch on sm_121. Finding 123's table numbers are
    withdrawn as one-start noise, and the table PR is dropped.** All on the overlay venv with the PLE stride fix applied.

    Kernel level, three sweep passes each, captured selections + synthetic decode shapes: the GB10 table entry is
    within 0.98–1.05× of the best swept config on every shape up to 512 rows (the only miss is a 2048-row
    decode-config cell at 1.22×, a geometry production never produces). Against the stock GB300 table pinned in the
    same harness: prefill chunks 1.01–1.03×, the 283-row tail 1.22–1.25×, decode 1×1 1.02–1.05×, 4×1 1.08–1.22×,
    4×4 **1.64–2.25×**, 16×4 1.30×, 32–256 rows 1.09–1.19×, 512 rows 1.00× — consistent with finding 120.

    Server level, cold, three starts per arm (medians): TTFT 7.5k 2.65 → 2.64 s, 29k 10.31 → 10.26 s, pairs
    8k+8k 5.32 → 5.29 s, 30k+8k 12.83 → 12.73 s, no-spec decode c=1 22.6 → 22.8, c=4 67.7 → 66.5, c=16
    148.0 → 148.5 tok/s. Every difference is inside the run-to-run band (single-stream decode alone spans
    22.4–24.7 across stock starts). Arithmetic agrees: the 4×4 verify shape saves ~45 µs × 12 layers ≈ 0.5 ms of a
    ~130 ms four-stream MTP step. Warm MTP n=3 on the fixed kernel, both tables: c=1 37.6–42.7 tok/s at 61–74 %
    acceptance, c=4 93–102, c=16 (five-wide) 95–99; **0 corrupted requests in 16 cells** — the multi-stream MTP
    numbers are quotable again.

    Fused pre-indexer: 18 µs at 3.8k tokens, 33 µs at 8k, tiles within 2 % of the best; decode scoring kernel 10 µs
    with 3 % headroom; `cooperative_topk` fails at launch on sm_121 (`launch_cooperative_cluster`), so the 12x gate is
    correct and `persistent_topk` stays. Conclusion: no GB10 profile on the QSA path pays at the server. The stacked
    prefill levers with real weight are the swizzled blockwise GEMM with larger chunks, the GDN autotune spaces, and
    the MoE grouped GEMM (31 % of prefill at 51 TFLOPS) — `stack1` measures the first two.

134. **Swizzled blockwise-FP8 GEMM inside the compiled server: the overlay must load the extension at import time and
    expose the kernel as a registered custom op, or every swizzle arm dies at startup.** `stack1`'s first swizzle arm
    (`SW_swz4k_1`, 22:32) failed in `determine_available_memory` with Dynamo's "Attempted to call function marked as
    skipped": the overlay called `torch.ops.load_library` lazily inside `apply_block_scaled_mm`, i.e. inside the traced
    forward. Two defects, both fixed in `stack_patch.py`: (a) the `.so` is now loaded at module import (the branch inside the
    forward is a module-level constant, which is what vLLM's compile freezes anyway); (b) `_C_swz2.blockwise_sm120` is
    registered only for CUDA (`STABLE_TORCH_LIBRARY`, no fake kernel), so it is wrapped as
    `torch.library.custom_op("fnswz::blockwise_sm120", mutates_args=("out",))` with a `register_fake`. Verified under
    `FakeTensorMode` before the restart. The stock-16k arm of the aborted run (22:17) measured 8k 2.47 s (2.57/2.47/2.47),
    30k 10.91 s, pairs 8k+8k 4.99 s, 30k+8k 13.43 s — consistent with gb10tune's stock arms. stack1 restarted 22:48 from
    the first arm. Not a finding about the kernel; the vLLM PR (#55180) integrates in C++ and has neither problem.

135. **stack1's first pass is INVALID for the 16k swizzle arm and crashed the FLA arm: all arms shared one torch.compile
    cache, and the cache key does not see an env-gated Python branch.** Sequence on the overlay venv (`stack1`, 22:48–23:35):
    stock-16k compiled the blockwise-FP8 wrapper with the swizzle branch False (key = batch 16384); swz-4k compiled it
    True (key = batch 4096); swz-16k then *hit the stock-16k graph* and never called the swizzled op — hence 2.50 / 10.95 s,
    identical to stock (finding 134's table); FLA-wide-4k hit the swz-4k graph and died at startup with
    `'_OpNamespace' 'fnswz' object has no attribute 'blockwise_sm120'` because that process had not registered the op.
    The swz-4k arm was a genuine stock re-run for a different reason: the extension's gate needs a ≥12 MiB activation
    slab and a 4k chunk at K=2560 is 10 MiB. Same class as `[[spec-compile-cache-key-omits-nspec]]`. Fix: every env-gated
    arm gets its own `FN_CACHE_ROOT` (`vllm-swz`, `vllm-fla`) in `stack1.sh`/`stack2.sh`; the poisoned shared
    `torch_compile_cache` (2.2 GB) was purged so the reverted-code arms (dv) cannot load the swizzle graph. stack1
    restarted 23:37 from the first arm; nothing from its first pass is quotable except the stock-16k numbers.

136. **Swizzled blockwise-FP8 GEMM at the server: −12 % TTFT at 30k with 16k chunks, null at 8k, null at 4k chunks;
    the wide FLA autotune space is null everywhere (`stack1`, two starts per arm after the finding-135 restart,
    `notes/data/stack1.txt`; overlay venv, no spec, prefix cache off; stock-4k reference = gb10tune's TB_stock_1..3).**

    | arm | 8k TTFT | 30k TTFT | 8k+8k wall | 30k+8k wall |
    | --- | --- | --- | --- | --- |
    | stock 4k (TB_stock, 3 starts) | 2.62–2.66 | 10.31–10.40 | 5.32–5.34 | 12.83–12.91 |
    | swizzle 4k | 2.65 / 2.63 | 10.41 / 10.29 | 5.34 / 5.29 | 13.08 / 12.81 |
    | FLA wide autotune 4k | 2.67 / 2.64 | 10.41 / 10.29 | 5.37 / 5.29 | 12.95 / 12.76 |
    | stock 16k | 2.59 / 2.47 | 10.81 / 10.84 | 5.00 / 4.93 | 13.35 / 13.33 |
    | **swizzle 16k** | 2.46 / 2.76 | **9.52 / 9.50** | 4.98 / 5.19 | **12.03 / 11.98** |

    Reading. (a) At 4k chunks the swizzle never engages: the extension gates on ≥12 MiB of activation and a 4k chunk at
    K=2560 is 10 MiB, so that arm is a stock re-run and reproduces TB_stock to the second. (b) At 16k chunks the gate
    opens for the weights above the 24 MiB L2 — the 36 GDN `in_proj_qkv` (25 MiB) and the 12 attention `q_proj`
    (30 MiB) — and the 30k prompt drops 10.82 → 9.51 s (−12 %, 6/6 requests within 20 ms); the 30k+8k pair drops
    13.34 → 12.0 s (−10 %). (c) At 8k the prompt is one 7.5k chunk, where the stock kernel is only mildly degraded
    (96 TF vs 150 standalone, finding 100), so the 3 % it should give is inside the 2.45–2.76 s start-to-start
    noise. (d) Combined with chunk size: swizzle-16k beats the best stock configuration (stock-4k, 10.3 s) by 8 % at
    30k and 6 % at 8k, i.e. the 16k chunk is now strictly better at every size, whereas stock-16k lost to stock-4k at
    30k (finding on prefill-batch-size). (e) The wider Triton autotune space for the GDN chunk kernels changes
    nothing at either size; the autotuner already sits at its optimum inside the vendored kernels, so the GDN lever is
    the fla-core fused intra-chunk kernel (finding 77), sized by the pr12 attribution. Third start for the swizzle =
    the profiled `stack2` run. PR #55180 server-level paragraph drafted in `notes/upstream/comment-55180-server.md`
    (not posted).


137. **The FlashInfer SM120 grouped MoE GEMM is latency-bound, not bandwidth- or compute-bound (`pr12`, ncu `--set full`
    on the standalone layer at M=7503, `notes/data/pr12.txt`, reports `pr12_gemm_{cold,warm}.ncu-rep` on the box).**
    GEMM1 4.19 ms / GEMM2 5.61 ms per layer (the 9.8 ms of finding 78). Both: grid = **48 CTAs, one per SM** (persistent),
    384 threads, **168 registers/thread and 89 KB smem, so the occupancy limit is 1 block by registers AND by smem**;
    achieved warps active 22.9 %; **issue slots busy 12 % (GEMM1) / 9.6 % (GEMM2); "No Eligible" 87.7 %; 2.6 active
    warps per scheduler, 0.16 eligible**; tensor pipe 28 %; L2 hit 74 % / 70 %; memory throughput 32–35 % of peak.
    Cold vs warm L2 across replay passes changes nothing (4.19 vs 4.22 ms). Reading: the SMs sit idle waiting on loads
    with too few warps to hide the latency; the tactic table is flat (finding 78) because every tile shares the same
    1-CTA/SM occupancy. What could move it: pipeline depth / L2 locality (the scheduler-swizzle experiment `pr12c`),
    or a kernel with two CTAs per SM (≤ 84 regs, ≤ 49 KB smem), i.e. a structural change in FlashInfer/CUTLASS, not
    a config. `dram__` metrics do not exist on GB10 (unified memory), so DRAM bytes are not measurable with ncu here.
    Decode shapes (`pr12b`, plain timing, random routing): M=1 141–144 µs per layer vs the 90 µs expert-byte floor
    (10 experts × 2.46 MB at 273 GB/s) = 1.6×; M=4 (the MTP-3 verify) 526–546 µs vs 360 µs (40 experts) = 1.5×;
    M=16 1,723–1,748 µs vs 1,440 µs = 1.2×; M=64 4,562 µs vs ~3.2 ms (≈360 distinct experts) = 1.4×; M=256 6,520 µs vs
    4.6 ms (all 512 experts, 1.26 GB) = 1.4× (`notes/data/pr12b.txt`). So at every decode shape the MoE sits 1.2–1.6× above
    its expert-byte floor, with the expand/finalize kernels inside that number.

138. **GDN chunked prefill: the whole fla-core gap is ONE kernel — the fused kkt+solve (`pr12`, per-kernel attribution,
    H=48, K=V=128, one sequence).** At T=7503 vendored 9,156 µs vs fla-core 8,245 µs (−10 %); at 29,263 37,076 vs
    32,605 (−12 %). `chunk_gated_delta_rule_fwd_kernel_h`, `chunk_fwd_kernel_o` and `recompute_w_u_fwd_kernel` are
    identical to the microsecond in both; the difference is vendored `chunk_scaled_dot_kkt_fwd_kernel` (876 µs) +
    `merge_16x16_to_64x64_inverse_kernel` (689 µs) = 1,565 µs against fla-core's single
    `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (704 µs). So the port is one Triton kernel plus the `chunk_size==64`
    dispatch in the vendored `chunk.py`; the wide autotune space (finding 136) was never going to find it. Worth on TTFT:
    the GDN share (~16 %) × 10 % ≈ 1.5–2 %. Small, but a clean sync PR.


139. **Grouped-GEMM tile-scheduler swizzle/raster: null at prefill and decode shapes, harmful at 29k (`pr12c`, FlashInfer
    0.6.17 `fused_moe_120` JIT-rebuilt from a copied csrc tree with env-driven `TileScheduler::Arguments`,
    `notes/data/pr12c.txt`; venv untouched, AOT module bypassed in-process).** Bit-identical checksums to the AOT module
    at every arm. M=7503: all eight (raster N/M × swizzle 1/2/4/8) within 52.4–53.2 TFLOPS. M=4: 528–534 µs, flat.
    M=29263: swizzle 1 → 75.1 (N) / 76.2 (M) TFLOPS, swizzle 2 → 73.9 / 72.6, **swizzle 4 and 8 → 64–65 (−14 %)**:
    grouping tiles for L2 reuse only makes the latency-bound kernel (finding 137) wait longer. Closes the L2/scheduler
    route for the MoE: the remaining lever is structural (a second CTA per SM, i.e. ≤ 84 registers and ≤ 49 KB smem,
    or a different grouped-GEMM design for 512 × ~150-row problems) and belongs to FlashInfer/CUTLASS. Issue text with
    the ncu evidence drafted in `notes/upstream/issue-flashinfer-sm120-grouped-gemm.md` (not posted). The JIT rebuild
    of the module takes 12 min at MAX_JOBS=4 on an idle box.


140. **Swizzle in the server, kernel-level (`stack2`, one profiled 30k prefill per arm, 16k chunks, `notes/data/stack2.txt`,
    traces on the box): the blockwise-FP8 GEMM kernel goes 2,532 → 1,011 ms over the request (192 calls, 13.2 → 5.3 ms
    per call, 2.5×), kernel-sum 12.23 → 9.48 s, profiled TTFT 13.08 → 9.83 s; third start for the swizzle arm.** The
    op attribution shows the swizzled path is the one that runs (`_C_swz2::blockwise_sm120` via `_fn_swz_mm`, 36+36+48+48+12
    calls at the GDN 16384-wide fused in-proj [M×2560 → 16384], the 6144-wide out-proj and the 13312-wide attention
    projection) and that stock's biggest single shape, [16384, 2560] × [2560, 16384] (the fused qkv+z, a 40 MiB weight),
    runs 27.9 ms per call stock vs 8.8 ms swizzled (3.2×). Everything else is unchanged within noise (QSA 1.17 → 1.08 s,
    hc 0.83 → 0.83 s, MoE grouped 1.91 → 1.61 s — the MoE difference is the same kernel finishing sooner once the FP8
    GEMMs stop thrashing the L2 around it, not a change in the MoE path). With stack1's 9.52 / 9.50 s this makes three
    starts at 30k: **10.82 → 9.5 s, −12 %**; the #55180 server paragraph (`notes/upstream/comment-55180-server.md`) is
    updated with these numbers and is ready to post on go.


141. **Warm agent turns are dominated by prefix-cache granularity, not by host overhead: the attention block size is forced
    to 1,600 tokens (1,568 without speculation) to match the Mamba state page, so the median turn recomputes 1,026 tokens it
    already had for 242 new ones (`turn`, overlay venv, prefix cache on, 16k chunks, one start per arm, `notes/data/turn.txt`,
    traces on the box).** (a) Regression over a 20k cached prefix + N fresh tokens (N = 16…4096, 3 each): MTP-3
    `TTFT = 485 ms + N / 2,613 tok/s`; no-spec `633 ms + N / 2,798 tok/s`. (b) The intercept is mostly recompute: the cache
    hits only at multiples of the block (19,200 of 20,076 tokens with spec; 18,816 without), so 876 / 1,260 tokens are
    prefilled again on every request — 335 / 450 ms of the intercept — leaving ~150–180 ms. (c) The profiled 16-token turn
    shows that remainder is GPU work too: kernel-sum 0.522 s over a 0.538 s span (97 % busy) for the 910-token recompute,
    i.e. 1.7k tok/s because at M≈900 the MoE grouped GEMM runs 3.2 + 1.8 ms per layer (finding 137's latency-bound kernel
    at ~18 rows per expert) = 46 % of the turn; true host + HTTP + template overhead is ~40 ms (tokenize + chat template of
    a 20k prompt: 24 ms). (d) Real trajectory replay, 23 warm turns: new tokens median 242, recomputed median 1,026 (spec) /
    1,080 (no-spec), TTFT median 0.59 s in both arms; recomputed/new ratio 2.65× / 3.4×. Mechanism (`platforms/interface.py`):
    `attn_block_size = align × ceil(mamba_page / (align × attn_bytes_per_token))`; in align mode the Mamba checkpoint block
    equals it; the log says "Setting attention block size to 1600 tokens to ensure that attention page size is >= mamba page
    size" and "Padding mamba page size by 0.25%". Consequences: (1) `disable_eagle_block_drop` (finding on #53388) was
    worth −26 % per warm turn because a dropped block is 1,600 tokens, not 16; (2) the KV budget of 4 GB is only 47 blocks
    (75,678 tokens), so LRU churn is coarse too. Levers: halve the Mamba page (fp8 SSM state → ~800-token blocks → ~−0.2 s
    per warm turn, quality to be checked by logprob divergence), or an allocator change upstream that lets one Mamba page
    span k attention pages so the attention block can stay small (≈ −0.4 s per warm turn, −65 %). The small-M MoE
    inefficiency is the second half of the same turn.


142. **Small prefix-cache block on the hybrid (`blk`, FN_KEEP_BLOCK overlay: keep `--block-size`, pad attention pages to the
    Mamba page; block 512 and 1024 vs the forced 1600; MTP-3, prefix cache on, 16 GB KV, one start each, `notes/data/blk.txt`):
    the overlay works and the regression intercept falls as designed, but the real-turn median does not follow.**
    **CORRECTION 2026-09-08: "MambaDType has no fp8, so this is the only in-config route" was wrong on the second half.**
    fp8 is indeed not a `MambaDType`, but `FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)` and the checkpoint
    ships float32, so `--mamba-ssm-cache-dtype bfloat16` halves the state and the derived block follows — without the
    padding this overlay pays, and with the per-step recurrent-state traffic halved as well. MiaAI-Lab measured the same
    switch on a Spark at 3,200 → 1,664 tokens, +6.8 % decode at 1 stream and +8.5 % at 8 (2026-09-06). Measured here as
    the `ssm` run.

    **Pre-registered prediction for that run, from the arithmetic rather than from their number.**
    `attn_block_size = align · cdiv(mamba_page, align · attn_page_1_token)`, and `attn_page_1_token` is derived from
    `kv_cache_dtype` (`cache_dtype == "auto"` → the model dtype, `platforms/interface.py:805`). Two consequences follow,
    and together they explain why their block is 3,200 and ours 1,600 on the same architecture: (a) halving the SSM state
    halves `mamba_page` and therefore the block — ours should go **1,600 → ~800**; (b) **fp8 KV DOUBLES the attention
    block**, because it halves the bytes per token in the denominator. They run `KV_CACHE_DTYPE=fp8` and we run bf16,
    which is the whole of the 2× difference, and their 3,200 → 1,664 is the two effects cancelling. The fp8-KV half is
    worth stating on its own: it doubles the pool's token capacity *and* doubles the prefix-cache block, so on a hybrid it
    buys long-context capacity by paying warm-turn recompute — their own report notes that at block 3,200 "prefix-cache
    hits are impossible below ~6,400 tokens of prompt", which is exactly this. That is a second cost for open RFC
    vllm#55196 ("fp8 KV gives little to no memory benefit on Mamba/GDN hybrid models"), which so far argues only from the
    mamba-page padding. Untested here; it is one `FN_KVDTYPE` value whenever the agent-turn harness runs again.

    **And a confound to name before the run, not after:** halving the mamba page does not only halve the block. The KV
    pool holds the same bytes but each page now covers half as many tokens with none of the padding, so capacity in
    tokens rises too — more retained prefixes across turns, and more room for concurrent requests (which is the same
    quantity vllm#55533 is about). A win in `ssm` is therefore attributable to *two* mechanisms of one knob: finer
    recompute granularity and a larger effective cache. The `MECH kv` line (GPU KV cache size) and the `schedwidth`
    cell separate them; the write-up must not attribute the whole effect to granularity by default.
    Server: "keeping attention block size 512 (derived minimum was 2048)"; KV capacity 348k tokens at 512 / 318k at 1024
    (vs 76k at 1600 with 4 GB, i.e. the padding costs far less than my per-block estimate — the QSA ring pages scale by
    block instead of padding). Generation sanity clean (acceptance 42–70 %, no garbage). Regression over the 20k cached
    prefix: 1600 → `485 ms + N/2613`; **512 → `265 ms + N/2728`** (108 instead of 876 tokens recomputed on the base);
    1024 → `444 ms + N/2679` (620 residual on that base). Trajectory replay, 23 warm turns, new tokens median 242:

    | block | recomputed tokens (median) | TTFT median | TTFT mean |
    | --- | --- | --- | --- |
    | 1600 (finding 141) | 1,026 | 0.592 s | 0.660 s |
    | 1024 | 953 | **0.547 s** | 0.667 s |
    | 512 | 628 | 0.631 s | 0.597 s |

    Reading: halving the recomputed tokens does not halve the turn because the remaining ~600-token prefill runs at the
    small-M rate (finding 141: ~1.7k tok/s at M≈900, MoE grouped GEMM latency-bound, finding 137) plus ~100 ms of extra
    per-turn cost at 3× the block count; the fixed 20k-prefix regression sees the full gain (−220 ms), real turns see
    −10 % on the mean at 512 and −8 % on the median at 1024, inside one-start noise. Verdict: the block lever is real but
    gated on small-M prefill efficiency; not worth an upstream proposal until the MoE small-M problem moves. The overlay
    (`tools/blk_patch.py`) is kept for that day.

143. **FLA fused kkt+solve port: correct and −7…−12 % on the whole GDN chunked forward (`flatest`/`flatest3`, branch
    `fla-fused-kkt-solve` in vllm-mambafix, `notes/data/flatest*.txt`).** Two porting traps, both mine: fla-core scales the
    gate cumsum by 1/ln2 and uses exp2, the vendored pipeline keeps the natural log and exp — with exp2 the A tensor was
    5e-2 off and the forward wrong; and fla-core hard-wires tf32 in the solve where the vendored `solve_tril` reads
    `FLA_TRIL_PRECISION` (default ieee) — the port now reads the same knob. With both fixed, A matches the two-kernel path
    within 1.95e-3 (a quarter bf16 ulp) and the full `chunk_gated_delta_rule` output within one bf16 ulp (7.8e-3 at |o|≈1.4)
    at T = 333, 2048, 5000, 7503, 16384, varlen included. Timing (H=16, HV=48, K=V=128): kkt+solve 394 → 189 µs at 2048,
    1,466 → 799 at 7503, 3,274 → 1,823 at 16384, 6,814 → 3,687 at 29263 (1.8–2.1×); whole forward −11.8 / −9.5 / −7.4 /
    −8.7 %. My first two test harnesses produced NaN and an illegal address by giving q the value-head count; q shares the
    16 key heads. PR body drafted (`notes/upstream/pr-fla-fused-kkt-solve.md`); opening it needs the user's go.


144. **MoE grouped GEMM, corrected: GEMM1 is at the DRAM floor, the "latency-bound" reading was ncu's DRAM percentage
    against the wrong peak; the contiguous-tile-run scheduler is bit-identical and −26…−29 %; the real headroom is GEMM2's
    write path and three elementwise kernels, including a finalize that is never fused on SM120 (`pr12d`, `moe_fin`,
    `notes/data/pr12d.txt`, `notes/data/moe_fin.txt`).** (a) Byte floor at M=7503: GEMM1 moves 0.84 GB of expert weights
    + 0.10 GB of fp4 activations + 0.19 GB of bf16 output = 1.13 GB, which is 4.13 ms at 273 GB/s; it measures 4.19 ms.
    Every scheduling experiment on it (tactics, swizzle, raster, cold/warm L2) was null because there is nothing left.
    GEMM2's floor is 0.82 GB = 3.0 ms; it measures 5.6 ms (~150 GB/s effective) — that is the only GEMM headroom, and it
    sits in the K=640 short loop + bf16 output write, not in scheduling. (b) The scheduler stall reason "sleeping" (mbarrier
    wait) is consumers waiting for DRAM-bound TMA loads; making each CTA take runs of 16 consecutive tiles (a permutation
    of the strided order, `tools/moe_sched_patch.py`) is bit-identical and **slower**: 53.9 → 38.5 TFLOPS at 7503,
    75.4 → 55.7 at 29263, 525 → 562 µs at M=4 — the strided order lets the 48 CTAs share one expert's weight tiles in L2,
    which is worth more than the per-tile tensormap switch. RUN=8/40 arms stopped as moot. (c) Per-kernel budget of one
    MoE layer call at M=7503 (torch profiler, `moe_fin.py`): grouped GEMMs 9.99 ms (2 launches), `finalizeMoeRoutingKernel`
    1.86, `doActivationKernel` 1.16, `expandInputRowsKernel` 0.99, routing/prefix 0.25 → 14.25 ms; at 29263: 21.6 + 7.15 +
    4.62 + 4.69 + 0.96 = 39.1 ms. `use_fused_finalize=True` and `False` give the same kernel list and times (14,251 vs
    14,143 µs): on SM120 the finalize is never fused into GEMM2's epilogue (`mayHaveFinalizeFused` is sm ≥ 90, but the
    SM120 block-scaled dispatch does not carry the FINALIZE epilogue), so GEMM2 writes 0.38 GB of bf16 that the finalize
    reads straight back, and GEMM1 writes 0.19 GB that the activation reads back. Fusing activation into GEMM1's epilogue
    and finalize into GEMM2's would remove ~1.1 GB of traffic ≈ 3 ms of the 14.25 (≈ 20 % of the MoE layer, ≈ 6 % of
    TTFT). That is the ask for FlashInfer (issue #4990 needs a correction: not occupancy, DRAM-bound + missing fusions;
    follow-up drafted in `notes/upstream/comment-fi-4990-correction.md`, not posted).

    **Correction (same day, finding 145):** the "never fused on SM120" statement in (c) was an artefact of my harness:
    FlashInfer's `profile_ids` is dead in 0.6.17 and, outside an `autotune()` context, the runner uses the fallback
    tactic (-1), which has no finalize fusion. The served vLLM path autotunes at warmup and the server traces show the
    fused-finalize GEMM2 (scatter epilogue) on every prefill layer with no `finalizeMoeRoutingKernel`; the decode buckets
    mostly pick the plain GEMM2 + finalize kernel. The activation-fusion ask stands; the finalize ask does not.


145. **A genuine FlashInfer MoE tactic sweep (through the autotuner, forcing one candidate at a time) shows the autotuner
    already finds the best GEMM2 tactic, and that every earlier "tactic sweep" on this box measured the fallback tactic
    (`tools/moe_fin4.py`, `notes/data/moe_fin4.txt`, M=7503).** Mechanics: `cutlass_fused_moe(..., profile_ids=[t1, t2])`
    accepts the argument and ignores it (the tactics come from `AutoTuner.choose_one`; outside `autotune(True)` that is the
    fallback tactic −1, i.e. the first config of each GEMM's list), so finding 78's "tactics 0–31 / 0–63 all within ±2 %"
    and today's `moe_fin2` were all the same fallback run. Forcing works only by monkeypatching `MoERunner.get_valid_tactics`
    (closure class, reached via `inspect.getclosurevars(module.cutlass_fused_moe)`) inside `autotune(True)`. GB10 has 20
    GEMM1 and 40 GEMM2 tactics (all occupancy 1). Results per layer call: fallback 14,070 µs (GEMM2 4.26 ms + separate
    finalize 1.82 ms); **autotuned 13,466 µs** (−4.3 %): GEMM2 with the fused-finalize scatter epilogue 5.12 ms, no finalize
    kernel; the best forced id (56, same epilogue) 13,485. The fused variants are interleaved in the list (rel 16, 32, 36 …),
    not a contiguous half; the swap-AB scatter variant is slower (6.5 ms). Numerics: the fused finalize changes the output
    checksum by −0.58 % (L1 sum 6.1368e11 → 6.1013e11) against the unfused path — the scatter accumulates the top-10
    partials with atomics in a different precision/order; that is the nondeterminism our PR #54948 gates, and it is also a
    small precision difference worth a logprob check on real prompts. Served prefill uses the fused variant (trace: 96 scatter
    GEMM2 launches per 30k prefill, zero finalize kernels); served decode buckets mostly use the plain GEMM2 + finalize kernel
    (2,736 plain vs 48 scatter launches in the c=1 trace). MoE lever status after this: GEMM1 at the DRAM floor (144),
    finalize fusion already on, activation fusion into GEMM1's epilogue (~1.2 ms/layer) is the one remaining kernel ask.


146. **The merged `weight > L2` swizzle gate is not Pareto-optimal, the activation-slab term should come
    back, and its small-M island is at M ≤ 1024, not 2048 (`swzM`, 2026-09-07, four starts,
    `notes/data/swzM.txt` + `notes/data/swzM_cd.txt`).** Standalone `_C_swzM` = the merged #55180
    dispatch with the swizzle exposed as an argument, so both orders are measured *at the same shape*
    rather than inferred across a gate. 6 shapes × M ∈ {1024…6144, step 512} = 66 cells,
    bit-identical on every cell and every arm.

    Gate scoring, medians of four starts, "regret" = summed % left on the table across all 66 cells:

    | gate | regret | worst cell |
    | --- | --- | --- |
    | never swizzle | 732.7 pp | 69.1 % |
    | **merged `weight > L2`** (what shipped) | **118.8 pp** | 9.9 % |
    | `+ (M ≤ 2560 ‖ A ≥ 14 MiB)` | 90.2 pp | 9.9 % |
    | slab only, `+ A ≥ 14 MiB` | 54.0 pp | 7.4 % |
    | `+ (M ≤ 2048 ‖ A ≥ 14 MiB)` (as proposed) | 48.6 pp | 8.7 % |
    | **`+ (M ≤ 1536 ‖ A ≥ 14 MiB)`** | **35.2 pp** | 5.8 % |
    | **`+ (M ≤ 1024 ‖ A ≥ 14 MiB)`** | **35.3 pp** | 5.8 % |

    (`A` = activation slab = M·K bytes, FP8.) **X = 1024 and 1536 are indistinguishable; 2048 is
    measurably worse; the 14 MiB knee from `bd84b180` survives unchanged.** The island is real —
    dropping it costs 54.0 vs 35.2 — so re-introducing the slab term that review removed is justified,
    but only with the island attached.

    **What the coarse grid had wrong.** The regression was recorded as an M=4096 band on 2560-wide
    weights. At M=4096 it is real and reproducible (16384×2560 −5.4 %, 12288×2560 −4.7 %,
    10240×2560 −3.0 %), but the *worst* reproducible losses are at **M=2560**, which the old grid
    never sampled: 5120×5120 −10.4 %, 10240×2560 −9.4 %, 16384×2560 −6.4 %.

    **The island is K-dependent, which is why X=2048 fails.** At K=2560 swizzle-8 wins at low M; at
    K=5120 the default order wins there, so an island reaching to 2048 takes reproducible losses of
    −8.5 % (7168×5120 M=2048) and −6.9 % (5120×5120 M=2048). Pulling X down to 1024 avoids them.

    **Noise, and why this needed four starts.** The winner flips between starts in **14 of 66 cells**;
    the default-order arm's start-to-start spread is median 4.7 % and **max 26.1 %**, while the
    swizzled arm is far steadier. At n=2 the X candidates scored 34.0 / 31.3 / 39.0 pp and could not be
    separated; at n=4 they are 35.3 / 35.2 / 48.6 and 2048 separates cleanly. The swizzled order being
    the *stable* one is itself an argument for it.

    **What must not be lost by any gate:** the large reproducible swizzle-8 wins, all at A ≳ 22 MiB —
    7168×5120 M=6144 **+223 %**, M=5632 +199 %, 14336×4096 M=6144 +158 %, 5120×5120 M=4608 +93 %.

    ⚠️ The harness and its 223 MiB CUTLASS tree live in an old session's `/tmp` scratchpad — the kind
    the 2026-09-03 reboot wiped. Move both to `/opt/llm/runners` before the next reboot or reproducing
    any of this costs a fresh CUTLASS clone.

    **Verification of the gate as committed (`swzG`, `notes/data/swzG.txt`):** bit-identical to the
    stock op on **66/66 cells, twice**; on the 26 cells where the two orders differ by >30 % the gated
    launch tracks the arm the predicate selects, **26/26**; the predicate's decisions score **37.0 pp**
    against the merged rule's 118.8 pp, matching the policy scoring. ⚠️ The runner's own performance
    check printed 141 pp and flagged 36 "gate picks the slower order" cells — **that estimator is
    invalid**: it sums `max(0, loss)` against arms timed in the same noisy run, so per-cell noise (up
    to 26 %) can only ever add, and it inferred which arm ran by comparing timings that are often
    within a few percent. Score a gate on medians across starts and on the deterministic predicate,
    never on same-run timing attribution.

    **Upstream: PR https://github.com/vllm-project/vllm/pull/55661 (2026-09-07).**


147. **Server A/B of the swizzle GATE: no measurable end-to-end effect. The one true control moves as much
    as the cells that changed (`swzab2`, 2026-09-07, six arms, `notes/data/swzab.txt`).** Both arms route
    the blockwise FP8 GEMM through the same standalone op via a `sitecustomize` patch, differing only in the
    swizzle the policy picks, so the gate is the single variable. `FN_BATCH=8192` so each prompt is one chunk
    and M is the prompt length; MTP off; UUID-prefixed prompts so the prefix cache never contributes; 5
    requests per size, first dropped; 3 starts per arm, interleaved.

    | prompt | tokens | gate picks | merged picks | merged median | gate median | gate faster |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1k | 1,066 | 1 | 8 | 0.529 s | 0.524 s | +0.95 % |
    | 2.5k | 2,596 | 1 | 8 | 1.135 s | 1.124 s | +0.97 % |
    | 3k | 3,102 | 1 | 8 | 1.291 s | 1.280 s | +0.85 % |
    | 4k | 4,125 | 1 | 8 | 1.582 s | 1.571 s | +0.70 % |
    | 5k | 5,143 | 1 | 8 | 1.869 s | 1.857 s | +0.64 % |
    | **7.5k** | 7,691 | **8** | **8** | 2.663 s | 2.638 s | **+0.94 %** |

    **7.5k is the control: both policies emit the identical launch, and it "improves" by +0.94 %** — as much
    as or more than every cell where the policies actually differ. The whole signal is a systematic offset
    (the gate arm always ran second in each pair), and the gate's true effect is inside noise of zero.
    Within-arm spread is 1.00–1.02×, so the harness is stable; it simply has nothing to resolve.

    ⚠️ Note 1k is **not** a control: at 1,066 prompt tokens the `M <= 1024` island does not apply, so the
    policies differ there too. Only 7.5k (activation slab 19.0 MiB ≥ 14) is identical-config.

    **Verdict against the pre-registered rule** (≥1–2 % at 2.5–5k with no losses → defend; <1 % → close):
    the changed region averages +0.79 % *before* subtracting a +0.94 % control offset, i.e. ≈0 after.
    **This is the answer to "likely not noticeable e2e": the reviewer was right.** The kernel-level 3–10 %
    on these GEMMs does not reach TTFT, because the blockwise FP8 GEMMs are only a fraction of prefill.

    Methodology note that nearly cost the whole run: the first attempt (`swzab`) produced four complete,
    plausible arms that measured **nothing** — `PYTHONPATH` pointed into `/tmp/claude-1000`, which is
    `drwx------ jschmied`, while the server runs as `uid=llm`, so `sitecustomize` was never importable and
    `site` swallows that error silently. My own check passed because I ran it as myself. The rerun added a
    hard gate: an arm without `SWZPATCH active` in its log aborts the run instead of emitting numbers.

    **Why the null was structural, not a sample-size problem (added after the A/B).** Splitting all 154
    swept cells by whether the slab gate changes the decision:

    | | cells | median \|Δ\| between the two orders | p90 | max |
    | --- | --- | --- | --- | --- |
    | the gate CHANGES the decision | 74 | **3.9 %** | 8.9 % | 11.0 % |
    | both policies agree | 80 | **18.8 %** | 173.5 % | 230.8 % |

    **The merged `weight > L2` rule already captures every large effect.** Every one of the top differences
    in the grid (+197 % to +231 %) sits in the agree bucket. The gate only ever operates where the two
    orders are within ~4 % of each other, capped at 11 % — so even a *perfect* gate over that region is
    worth at most ~11 % on the worst single GEMM shape, on a kernel that is a fraction of prefill. A null
    at the server is what that predicts, and no amount of extra starts would change it. This closes the
    question for both the three-condition and the simplified two-condition form.

    **Where the value actually is:** the `weight > L2` boundary itself, which is in the agree bucket and
    which both policies get wrong — `6144x4096` weighs exactly `l2CacheSize` (25,165,824 B) so the
    condition is false by one byte, and at M=6144 that costs **138 %** (69.5 vs 165.6 TF). That is a
    separate, better-motivated change against merged code.


148. **The L2 boundary sweep says the merged gate tests the wrong variable: across 232 cells from three
    sweeps, `weight > L2` leaves 939.6 pp on the table and the activation slab alone leaves 137.3
    (`swzL2`, 2026-09-07, 3 starts, 87 new cells, `notes/data/swzL2.txt`).** Ratios weight/L2 = 0.70 →
    1.71 at K ∈ {2560, 4096, 5120}, M ∈ {2048, 4096, 6144}, both orders forced at each shape,
    bit-identical everywhere.

    Grouping the new cells by activation slab **A = M·K and ignoring weight/L2 entirely**:

    | A (MiB) | 5 | 8 | 10 | 15 | 16 | 20 | 24 | 30 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | median sw8 gain | +1.6 % | −5.5 % | −4.6 % | **+7.2 %** | +4.8 % | **+38.0 %** | **+124.3 %** | **+68.4 %** |

    A clean sign flip between 10 and 15 MiB — the same knee as `bd84b180`'s 14 MiB — and the weight/L2
    ratio does not order the data at all. Scored over the union of all three sweeps:

    | gate | all 232 cells | the 168 with weight > L2 | the 64 with weight ≤ L2 |
    | --- | --- | --- | --- |
    | **merged `weight > L2`** | 939.6 pp, 101 bad | 279.0, 66 | **660.6, 35** |
    | **slab only `M·K ≥ 14 MiB`** | **137.3 pp, 36 bad** | **96.6, 25** | **40.7, 11** |
    | `weight > L2` **and** slab | 757.2, 60 | 96.6, 25 | 660.6, 35 |
    | always swizzle | 410.3, 94 | 279.0, 66 | 131.3, 28 |

    Slab-only is **6.8× better overall and better in both halves**. The L2 term contributes nothing above
    the boundary and does real damage below it: it vetoes the swizzle on sub-L2 weights with large
    activations, where the swizzle wins up to **+150 %** (5632×4096, weight 0.92×L2, M=6144: 95.7 %).

    **Why three of us missed this.** Both earlier sweeps chose shapes with weight > L2 — the tuning set has
    no sub-L2 shape at all — so the condition was constant-true and therefore untestable. It looked
    necessary because it never varied. My own #55661 kept it as a precondition, which is why that gate
    could not reach the sub-L2 cells either.

    ⚠️ **Slab-only is not the final answer.** Counter-example in finding 137: `2560×6144` (weight 15 MiB,
    0.62×L2, narrow N) has the *default* order winning 1–8 % at every M from 2048 to 16384, with A from
    24 to 96 MiB. N is only 2560 there — few column tiles to reorder — so the raster likely needs a
    minimum tile count, i.e. N belongs in the predictor. **Do not open a PR on slab-only before a sweep
    that varies N at fixed K and A.** That is the same trap #55661 was closed for.

149. **N is the missing term, and it is a conjunction with the activation slab: swizzle iff A ≥ 14 MiB
    AND N ≥ ~3840. Over 368 cells from four sweeps the merged `weight > L2` rule leaves 1,263 pp and this
    leaves 218 (`swzN`, 2026-09-07, 3 starts × 160 cells, `notes/data/swzN.txt`).** Grid designed so every
    variable moves independently: N ∈ 1280…20480 (16×), K ∈ {2560, 4096, 5120, 6144}, M ∈ 2048…8192,
    weight/L2 from 0.13 to 5.0. Bit-identical on all 480 measurements, 0 skipped.

    Median sw8 gain, rows = N, columns = A = M·K (MiB), pooled over K:

    | N \ A | 5 | 10 | 15 | 20 | 24 | 30 | 40 | 48 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1,280 | +0 | +1 | +1 | +1 | −1 | −1 | −3 | −5 |
    | 2,560 | +2 | −1 | −7 | −8 | −6 | −6 | −4 | −5 |
    | 3,840 | +4 | −9 | −8 | +2 | +12 | +18 | +15 | +49 |
    | 5,120 | +7 | −9 | −8 | +14 | +54 | +60 | +115 | +154 |
    | 7,168 | +0 | −7 | +9 | +32 | +129 | +230 | +210 | +137 |
    | 10,240 | +4 | −5 | +5 | +55 | +122 | +213 | +227 | +232 |
    | 20,480 | −3 | −3 | +7 | +49 | +133 | +221 | +230 | +232 |

    **At N ≤ 2,560 the swizzle never wins at any activation size** (median −1.9 % and −5.8 % in the
    A ≥ 20 MiB band) — the `2560×6144` counter-example of finding 148 was not a quirk, it is the whole
    narrow-N region. **Below A ≈ 14 MiB it never wins at any N.** Both conditions are necessary; neither
    is sufficient. Mechanically that fits: the raster reorders CTAs to reuse the weight across the N
    dimension, and with few column tiles there is nothing to reorder.

    Fitted on the N sweep alone, then tested on the three earlier sweeps untouched:

    | gate | N-sweep (fit) | tuning | held-out | boundary | **all 368** |
    | --- | --- | --- | --- | --- | --- |
    | merged `weight > L2` | 363.6 | 118.8 | 299.1 | 598.5 | **1,263.2 pp, 138 bad** |
    | slab only, A ≥ 14 MiB | 199.5 | 54.0 | 57.1 | 27.3 | 324.9 pp, 79 bad |
    | **A ≥ 14 MiB and N ≥ 3,840** | **90.3** | 54.0 | 57.1 | 29.3 | **217.7 pp, 59 bad** |
    | always swizzle | 379.8 | 118.8 | 149.5 | 161.5 | 750.1 pp |

    The N term costs nothing where it was not fitted (tuning and held-out are unchanged to the decimal,
    boundary 27.3 → 29.3) and halves the regret where narrow N actually occurs. **No L2 term appears in
    the winning rule at all.**

    ⚠️ Two things before this becomes a PR. (1) **Portability is now a bigger question, not a smaller
    one**: dropping `weight > L2` means the swizzle would activate on parts where it currently never
    fires (96–128 MiB L2), which we cannot test. Expressing the slab threshold as ≈0.58 × L2 would at
    least keep one term hardware-derived. (2) **The e2e lesson from #55661 still applies** — but the
    differences here are up to **+233 %**, two orders of magnitude larger than the 4–11 % that gate
    argued over, and our own model cannot test it (K=2560, N=16384 → weight 40 MiB > L2, where merged
    already picks correctly).

150. **Mechanism for findings 148–149: the gate tests total weight, but what overflows the L2 is the
    co-resident CTA wave's working set — and that is set by SM count, tile N and K, none of which the
    gate looks at (2026-09-07, arithmetic; tile shape verified from the dispatch header).**
    `scaled_mm_blockwise_sm120_fp8_dispatch.cuh:142` — `TileShape = Shape<_128, _128, _128>` (and
    `<_64,_128,_128>` for the swap-AB path), so **tileN = 128 in both**; GB10 reports 48 SMs and
    25,165,824 B of L2.

    With one CTA per SM, an N-major raster puts **48 distinct N-tiles in flight at once**, each needing
    its own K-length weight column. Instantaneous weight working set = `48 × 128 × K`:

    | K | linear raster | vs L2 | swizzle-8 | vs L2 |
    | --- | --- | --- | --- | --- |
    | 2560 | 15.0 MiB | 0.62× | 2.5 MiB | 0.10× |
    | **4096** | **24.0 MiB** | **1.00×** | 4.0 MiB | 0.17× |
    | 5120 | 30.0 MiB | 1.25× | 5.0 MiB | 0.21× |
    | 6144 | 36.0 MiB | 1.50× | 6.0 MiB | 0.25× |

    The swizzle does not make the weight fit; it shrinks the **concurrent** footprint ~6× by making
    co-resident CTAs share tiles. This accounts for both empirical terms of finding 149 with **no fitted
    constants**:

    - **The N term is a tile-count condition.** N=1280 → 10 tiles, N=2560 → 20 — both fewer than the 48
      SMs, so the wave cannot spread that wide, wraps in M instead, and its footprint is already small:
      nothing for the swizzle to fix. N=7168 → 56 ≥ 48, full-width wave, biggest win. Exactly the shape
      of the measured table (N ≤ 2560 never wins; N ≥ 7168 wins +130…230 %).
    - **K=4096 is violent because its linear wave working set is 24.0 MiB — identical to L2.** That is a
      *second* quantity landing on the same number as `6144×4096`'s total weight, and it is the one that
      matters. The coincidence at that shape is arithmetic, not luck.

    A principled gate would therefore be ≈ `sm_count × tileN × K > l2CacheSize` **and**
    `N / tileN ≳ sm_count` **and** enough M work — every term hardware-derived, which also answers the
    portability objection that sank #55661's fitted constants.

    Also note L2 is shared with the activation tiles, the output writes and any concurrent kernel or
    co-tenant, so the linear raster overflows it **earlier** than this arithmetic says, never later —
    the "does the weight fit" framing is a sequential question asked of a parallel machine.

    **Untested prediction, and the way to falsify it:** the crossover should move with `sm_count` and
    `tileN`, not with total weight. Forcing the 64×128×128 tile (swap-AB path) should shift the K at
    which the flip occurs by 1×, and capping CTAs per launch should shift it linearly. Neither is done.


151. **Draft-vocabulary size, decided from OUR distribution instead of the field's: the whole observed vocabulary is
    48,476 ids, so a 65k slice does not exist here, and the coverage curve saturates so early that the interesting
    range is 4k–16k, not 65k (`build_vocab.py` on 86 SWE-bench trajectories + 14,324 local source/doc files,
    `notes/data/dvsize-coverage.txt`, CPU only).** MiaAI-Lab shipped a 65,536-row draft head and argued 65k over 32k
    because "the crossover where lost acceptance eats the byte saving is around 88–90 % coverage, and 65k costs 3 points
    of byte saving to buy 3.6 points of coverage" — built from 513 MiB of wikitext-103 + 47 MiB of Python + model output,
    160M occurrences over 104,522 distinct ids. Rebuilt on our corpus (158,275,265 weighted occurrences: assistant output
    ×20, tool output ×1, code ×3; held-out = 10 trajectories by id hash):

    | slice | train coverage | held-out assistant coverage |
    | --- | --- | --- |
    | 1,024 | 75.58 % | 75.32 % |
    | 2,048 | 83.78 % | 84.53 % |
    | **4,096** | 90.84 % | **90.79 %** |
    | 8,192 | 95.91 % | 95.19 % |
    | 16,384 | 98.80 % | 98.05 % |
    | 32,768 | 99.88 % | 99.63 % |
    | 65,536 (= all 48,476) | 100.00 % | 99.96 % |

    Two things follow. (a) **Their 65k recommendation does not transfer**: our corpus contains only 48,476 distinct ids
    in total, and the held-out *assistant output* — the distribution the drafter actually proposes into — has just
    **2,048 distinct ids over 88,845 occurrences**. Agent traffic is far narrower than wikitext, so the curve saturates
    at a tenth of their size. (b) Applying *their own* crossover rule to *our* curve puts the optimum at **~4,096 rows**
    (90.79 % held-out), not at our shipped 32,768 (99.63 %) — 4k saves 98.4 % of the head's bytes against 86.8 % at 32k,
    and det-135's +6.4–6.8 % at c=1 was measured at the conservative end of the range. The sizes below 8k are the ones
    nobody has measured, on either project.

    Method note, recorded because it costs a baseline: this rebuild **overwrote** the 8k/16k/32k files that det-135 was
    measured with (`/opt/llm/runners/dv/draft_vocab_*.txt`, 2026-09-08 21:17). The corpus has grown since, so the new 32k
    set is not byte-identical to the old one. A size sweep must therefore re-measure 32k as its own arm rather than reuse
    det-135's number as the baseline.


152. **We have been running `--async-scheduling` with MTP for weeks, in every benchmark and in prod, while our own
    launcher carried a comment saying never to do it (code read 2026-09-08, `config/scheduler.py:179`,
    `config/vllm.py:1270-1319`, `v1/executor/multiproc_executor.py:558`).** `SchedulerConfig.async_scheduling` defaults
    to **`None`**, not `False`, and the `None` branch *enables* it unless something incompatible is present. MTP is in
    `EagleModelTypes`, so the speculative-decoding exclusion does not apply to us; `MultiprocExecutor.supports_async_scheduling()`
    returns `True`, so the executor exclusion does not either; we do not set `disable_padded_drafter_batch`. Every branch
    falls through to `self.scheduler_config.async_scheduling = True`. We never pass `--no-async-scheduling`.

    The comment in `serve-fnmain.sh` reads: "NEVER combine MTP with `--async-scheduling`: `_prepare_ngram_context` reads
    the CPU token mirror while it still holds speculation's -1 placeholders, giving a wrong n-gram context silently. No
    benchmark reveals it." Two things are wrong with it now. **First, `_prepare_ngram_context` is not about n-gram
    speculation at all** — it builds the PLE table's n-gram *embedding* context (`uses_ngram_embedding`,
    `models/qwen4_exp/nvidia/model_state.py:65`), so it runs for every request on this model whatever the speculation
    method, and it reads `req_states.all_token_ids.gpu`, a device tensor, not a CPU mirror. **Second, and worse, the
    caution was never acted on**: a warning that the launcher does not enforce and the default overrides is not a
    safeguard, it is a note that made us believe we were safe.

    Evidence that the hazard is at least not firing in the sequential path: det-181 and det-184 get bit-exact greedy
    reproduction across restarts with all four determinism fixes — with async scheduling on the whole time. That is
    strong for c=1 and says nothing about c>1. The field runs the same combination deliberately: MiaAI-Lab's dual-Spark
    tuning report keeps `--async-scheduling` for its best c=4 aggregate (140.7 vs 136.2 tok/s) with no accuracy note.

    **Queued as `asched`:** `--no-async-scheduling` against the default, deterministic stack on, cross-arm output
    comparison at c=1 and c=8. If the two arms agree bit-for-bit the caution is dead and the speed is ours to keep; if
    they diverge at c>1 only, that is a real defect in a configuration vLLM enables by default for every MTP user on a
    hybrid, and it belongs upstream. Either way the launcher comment gets replaced by a flag, because a comment is not
    a control.


153. **`--mamba-ssm-cache-dtype bfloat16`: the block halves exactly as predicted, agent turns get 9.6 % cheaper in
    total, the *median* turn gets slightly worse, and it costs a 5.1 % change in the model's own top-1 predictions
    (`ssm`, six arms interleaved bf16/fp32 × 3 starts, `notes/data/ssm.txt`, paired table
    `notes/data/ssm-paired-turns.txt`).** MTP-3, prefix caching on, `FN_MAXLEN=32768`, `FN_BATCH=4096`, util 0.75,
    deterministic stack on in every arm.

    **Mechanism — both halves pre-registered before the run, both confirmed.** Block **1,600 → 832** (predicted "~800"
    from `attn_block_size = align · cdiv(mamba_page, align · attn_page_1_token)`, not from the field's number); mamba
    padding 0.25 % → 0.48 %; and the second, less obvious half: the padding leaves with the page, so **KV capacity rises
    too** — 175–182k tokens at fp32 against 214–247k at bf16, max concurrency 5.4× → 6.5–7.5×. The fixed-prefix
    regression over a 20k cached prefix does exactly what the mechanism says: intercept **559/561/565 ms → 275/274/271 ms
    (−51 %)** with the prefill *rate* untouched (2462–2477 vs 2446–2470 tok/s). Three starts, ranges given, no overlap.

    **Real agent turns are the cell that decides it, and they say something the intercept does not.** Trajectory replay,
    24 warm turns, median 262 new tokens, paired turn-by-turn across arms:

    | | fp32 | bf16 |
    | --- | --- | --- |
    | recomputed tokens / turn (median) | 1,042 | 820 |
    | prefix-cache hit rate over the replay | 81.8 % | 86.7 % |
    | TTFT **median** | 0.663 / 0.665 / 0.675 s | 0.688 / 0.752 / 0.695 s |
    | TTFT **mean** | 0.779 / 0.796 / 0.795 s | 0.713 / 0.722 / 0.707 s |
    | **total over the 24 turns** | **18.84 s** | **17.04 s (−9.6 %)** |

    The median is *worse* and the mean is *better*, and both differences are outside the three-start spread — the arms
    reproduce to a few ms per turn (turn 18: 0.526/0.528/0.521 vs 1.161/1.166/1.167). bf16 is faster on **15 of 24**
    turns, and the paired table shows why: the win tracks the recompute exactly. Where the smaller block leaves less to
    re-prefill it wins large (turn 12: 788 vs 1,812 tokens, −657 ms; turn 18: 659 vs 1,747, −640 ms; turn 24: 1,771 vs
    2,923, −502 ms), and where the boundary happens to land worse it loses (turn 8: 979 vs 339, +458 ms; turn 13: 1,180
    vs 604, +465 ms). Recompute per turn is `new_tokens + (tokens since the last boundary)`, which is a modular lottery
    per turn; halving the block halves its *expectation* and its worst case, not every draw. **So this is a tail lever:
    it does not make the typical turn faster, it removes the expensive ones**, which is what accumulates over a session.
    Finding 142 measured the same knob through a padding overlay and called it flat — it was looking at the median.

    **What it does NOT buy.** Decode at c=1 is unmoved (bf16 19.3–19.5 / 17.7–17.8 / 19.4–19.5 tok/s against fp32
    19.6–19.7 / 16.4–16.8 / 19.7–19.9, rep-for-rep, rep 1 being the only cell bf16 wins) — MiaAI-Lab's **+6.8 % at one
    stream does not reproduce here**. MTP acceptance is unchanged (52/42/49 % vs 55/38/54 %). Aggregate at c=8 is
    26.7–28.3 vs 25.5–27.0 tok/s, inside the spread.

    **It is not free, and a needle test cannot see the price.** Both arms are internally bit-exact (0 disagreeing
    positions over 8 greedy repeats, deterministic stack on), so the arms are cleanly comparable — and across arms
    **127 of 2,504 positions (5.1 %) differ in their modal top-1 token**, max |Δ mean forced logprob| 5.20, first at
    position 36. For scale, the four determinism defects we spent a week removing moved 110/2,504 (det-184). Halving the
    recurrent state's precision perturbs this model's predictions by the same order of magnitude. That is a *deterministic*
    change rather than a nondeterministic one, and whether it costs task quality needs a task eval, not a logprob count —
    but "needles 15/15 unchanged", the evidence the field shipped it on, would never have detected it.

    **Verdict: a real −9.6 % on agent-turn time with a real precision cost, not a free lever.** Prod adoption is one
    `FN_SSM_DTYPE` line and is the user's call; it should be paired with a task-level quality check first.

    **Free by-product: vllm#55533 does not reproduce here.** `schedwidth` at c=8 shows `num_requests_running` median 5
    and max 8 in *every* arm, both block sizes — the scheduler reaches the full batch. The 1+k mamba-block charge is real
    in the code, but our pool (176–247k tokens) is far from the ceiling that produces the reported {2,2,2} window.


154. **FLA fused kkt+solve at the server: −1.4 % cold TTFT at 8k and −1.1 % at 30k, reproducibly; null on warm agent
    turns; and its decode "win" is not a speed effect at all but a 1-ulp numerics change flipping MTP acceptance by up
    to 10 pp in either direction (`pstack`, six arms interleaved fla/base × 3 starts, `notes/data/pstack.txt`).**
    Overlay gate clean in every arm (4 marker hits on `fla`, 0 on `base`); overlay verified reverted byte-clean at the
    end, which matters because that venv serves prod.

    | cell | base (3 starts) | fla (3 starts) |
    | --- | --- | --- |
    | cold TTFT 8k (7,528 tok) | 3.187 / 3.196 / 3.321 s | **3.144 / 3.150 / 3.152 s** |
    | cold TTFT 30k (29,288 tok) | 12.116 / 12.207 / 12.215 s | **12.021 / 12.055 / 12.091 s** |
    | warm-prefix intercept | 544 / 551 / 557 ms | 543 / 550 / 551 ms |
    | 24-turn replay total | 18.67 / 18.77 / 18.83 s | 18.74 / 18.67 / 19.58 s |

    The cold-prefill cells are the only ones that move, and they move cleanly: **the arms' ranges do not overlap on
    either**, across three independent server starts. −1.4 % / −1.1 % is exactly the arithmetic finding 143 implies —
    a kernel that is 1.8–2.1× faster and worth −7…−12 % of the GDN chunked forward, applied to a GDN scan that is
    around a tenth of prefill. The warm-prefix intercept and the real agent replay are both null (−0.21 % over 24
    paired turns, inside the spread; one `fla` start totalled 19.58 s against 18.67 s for another).

    **The decode result is the one worth reading carefully, because it looked like a win.** Rep-for-rep at c=1, `fla`
    beat `base` by 5.2 % on prompt 0 and 10.3 % on prompt 1, and *lost* 2.7 % on prompt 2 — every range tight and
    non-overlapping across the three starts. It is not a kernel-speed effect. The tok/s tracks acceptance exactly:

    | prompt | base acceptance / AL | fla acceptance / AL | Δ tok/s |
    | --- | --- | --- | --- |
    | 0 | 54.8 % / 2.64 | 61.4 % / 2.84 | +5.2 % |
    | 1 | 38.4 % / 2.15 | 48.8 % / 2.46 | +10.3 % |
    | 2 | 54.4 % / 2.63 | 50.6 % / 2.52 | −2.7 % |

    Acceptance is **identical to the decimal across all three server starts within each arm** (61.4 / 61.4 / 61.4) — the
    deterministic stack is doing its job, so these are not noisy measurements. The fused kernel matches the two-kernel
    path to within one bf16 ulp (finding 143) but is not bit-identical, and that is enough to change which draft tokens
    the target accepts. On two prompts it helps, on one it hurts. **Three samples of a lottery whose expectation is
    zero, not a lever.**

    **Two consequences.** (1) Verdict on finding 143: the port is correct and worth ~1 % of cold prefill, which is real
    but does not move agent turns; per the pre-registration, that closes it as "correct, kernel-level win, not
    measurable end to end" rather than something to ship here. The upstream PR draft still has merit *as a kernel PR*
    (1.8–2.1× on the kernel, one-ulp equivalence, both porting traps documented) and should be judged on that, not on
    an end-to-end number it was never going to produce. (2) The general caution is bigger than this run: **MTP
    acceptance on this model is chaotic with respect to numerical changes at the ulp level.** A ±10 pp swing from a
    one-ulp kernel difference means acceptance figures are not comparable across builds that differ numerically at all —
    ours, the field's, or upstream's — unless the arms are bit-identical. That retroactively explains a good deal of the
    acceptance scatter in `the-field.md`, and it is why every acceptance comparison from here needs the deterministic
    stack on *and* an explicit statement of whether the arms are bit-identical.


155. **MTP `num_speculative_tokens=4` is WORSE than our shipped 3 here (−3.4 % at c=1), the field's +11.4 % does not
    reproduce, and vllm#55533's scheduler collapse does not reproduce either (`mtp42`, n3/n4 interleaved × 3 starts
    plus one no-spec arm, `notes/data/mtp4.txt`).** k=4 was the one untested cell on our own list: our sweep stopped at
    3 because k=5 hard-fails on the block-size hole (compress_ratio 4 ⇒ n = 0…4 and 9…12 legal, 5…8 not), which made 5
    look like a wall rather than a gap. MiaAI-Lab reported the k-curve monotonic 0→4 on a dual-Spark pair, +11.4 % from
    3 to 4.

    **c=1 on real held-out agent prompts, three server starts per arm, per prompt:**

    | prompt | n3 tok/s (×3) | n3 accept / AL | n4 tok/s (×3) | n4 accept / AL |
    | --- | --- | --- | --- | --- |
    | 0 | 18.9 / 19.0 / 19.0 | 53.7 % / 2.61 | **19.5 / 19.6 / 19.4** | 53.1 % / 3.12 |
    | 1 | 16.7 / 16.8 / 16.6 | 38.4 % / 2.15 | **17.0 / 16.9 / 16.8** | 36.6 % / 2.46 |
    | 2 | **19.8 / 19.8 / 19.7** | 53.2 % / 2.60 | 17.4 / 17.2 / 17.2 | 35.7 % / 2.43 |
    | Σ of per-prompt medians | **55.5** | | 53.6 (**−3.4 %**) | |

    Acceptance is identical to the decimal across all three starts within each arm — the deterministic stack again — so
    these are not noisy cells. The shape is the trade the run was designed to expose: **k=4 buys draft length and loses
    acceptance rate** (AL 2.61 → 3.12 but 53.7 → 53.1 %; 2.15 → 2.46 but 38.4 → 36.6 %), and on prompt 2 the rate
    collapses 53.2 → 35.7 % and takes 13 % of the throughput with it. Two prompts gain ~2–3 %, one loses 13 %; net
    negative. At c=8 the ranges overlap with n3 ahead on two of three pairs (n3 25.3 / 25.2 / 23.4 against n4 21.9 /
    22.5 / 25.2). **k=4 is not a prod change.** Their result is not refuted — a dual-Spark TP=2 pair with a different
    checkpoint is a different machine — but it does not transfer, and it is the second field number tonight that did
    not (see finding 153 on the bf16 decode claim).

    **vllm#55533 does not reproduce, and the run says so on both of its pre-committed conditions.** (a) The scheduler
    is not pinned near 3: `num_requests_running` at c=8 has median 5–6 and reaches **8** in every arm (15–21 samples at
    width 8), with max waiting 6. (b) The decisive half fails outright — the no-spec arm manages **19.9 tok/s** at c=8
    against 23.4–25.3 for n3, so MTP is not "2-3× slower than no-spec above bs=3" here, it is comfortably faster.
    What *is* visible is the mechanism behind their formula, in the block size itself: **1,568 tokens at k=0, 1,600 at
    k=3, 1,616 at k=4**, and the KV pool falls 180,224 → 167,401 tokens from k=3 to k=4. The `1 + k` mamba-block charge
    is real; our pool is simply far from the ceiling that produces their {2,2,2} window.

    **One thing worth flagging against our own numbers: the KV pool is not a stable quantity per config on this box.**
    The three n3 starts, identical configuration, sized at **180,224 / 119,369 / 171,641 tokens** — a 34 % spread — and
    the small start is exactly the one whose c=8 histogram never reached width 8 (max 6). So capacity *does* gate
    concurrency when the pool comes up small, which is a partial corroboration of #55533's mechanism arriving through
    allocation variance rather than through k. Finding 153's KV comparison survives this (bf16 214–247k against fp32
    176–182k, non-overlapping), but any future single-start KV number should be treated as one draw, not as the config.


156. **CPU idle states cost 0.7–0.9 % of a decode step on GB10, not the 5–6.6 % the field measured (`cstates`, ONE
    server, six cells interleaved ON/OFF with a live `cpupower` toggle and no restarts, `notes/data/cstate.txt`).**
    The mechanism check is in every cell: `/sys/…/cpuidle/state*/disable` reads `1 1 1 1` in the OFF cells and
    `0 0 0 0` in the ON cells, and the states are verified restored at the end (they are — leaving them off would have
    contaminated every later run). This is the cleanest A/B available to us: one process, one KV pool, one compile
    cache, the only thing changing is a host setting toggled between cells.

    **Cell 1 must be discarded and is why the design mattered.** `on_1` reads 19.1 / 16.8 / 19.8 tok/s against
    26.4–26.8 / 21.6–21.9 / 26.5–26.8 for every later cell: it paid the cold prefix cache. Comparing it to anything
    would have manufactured a 40 % "effect" from cache state. The five warm cells, per prompt:

    | prompt | idle states ON | idle states OFF | Δ |
    | --- | --- | --- | --- |
    | 0 | 26.4 / 26.6 | 26.7 / 26.7 / 26.8 | +0.75 % |
    | 1 | 21.6 / 21.7 | 21.7 / 21.8 / 21.9 | +0.7 % |
    | 2 | 26.5 / 26.5 | 26.6 / 26.7 / 26.8 | +0.9 % |

    Same sign on all three prompts with barely-overlapping ranges, so the effect is real and it is **under 1 %**.
    Cold TTFT is untouched (8k: ON 3.177–3.211 s, OFF 3.189–3.196 s; 30k: ON 12.208–12.255 s, OFF 12.193–12.216 s).

    MiaAI-Lab measured +5–6.6 % on a dual-Spark pair by the same live-ablation method, so this is the **third field
    number tonight that did not transfer** (after the bf16 decode claim in 153 and the k=4 curve in 155). The mechanism
    is plausible here — this box's LPI-1 has a 42 µs exit latency and our c=1 MTP-3 step issues ~2,636 kernel launches
    — but the measured cost is an eighth of theirs. **Verdict: not worth taking.** It is a host-wide setting that needs
    an `@reboot` job to survive a power cycle and raises idle power on a box with a PD limit, and 0.8 % does not pay
    for that.

157. **`ishare` voided itself on a shell bug in its own gate, not on the flag — and the flag had engaged correctly
    (`notes/data/ishare-void.txt`).** The gate read
    `MECH: $(grep -aoE "index_share…" $LOG | head -1 || echo NOT-IN-LOG)`. In a pipeline the exit status is **`head`'s**,
    which is always 0, so the `||` fallback never fired, both arms reported an empty marker, and the second `case` fell
    through to its `*)` branch and declared "the base arm ALSO has the flag on". The run stopped after two of six arms.
    The logs say the opposite: `ish_1` carries
    `speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3, 'disable_eagle_block_drop': True,
    'index_share_for_mtp_iteration': True}` and `base_1` carries no such key.

    This is the **fifth void run**, and the first voided by the *gate* rather than by a knob that could not reach the
    cell (`cgsize2`, `cgnone2` c≥4, `thr`, `vpp4` were all the other kind). New rule in `method.md`: a mechanism check
    must be tested as a captured **string**, never as a pipeline's exit status. Re-queued as `ishare2` with the fix.

    The two arms that did run suggest the answer anyway, and it is not the field's: one start each, c=1, per prompt —
    `ish` 19.2 / 16.8 / 19.9 tok/s at acceptance 54.8 / 39.1 / 53.2 % against `base` 19.1 / 16.6 / 19.9 at
    54.8 / 38.4 / 54.4 %. Accept length is 2.64 / 2.17 / 2.60 versus 2.64 / 2.15 / 2.63 — MiaAI-Lab's 2.42 → 2.52 gain
    does not appear. One start is not a result; `ishare2` will say properly.
