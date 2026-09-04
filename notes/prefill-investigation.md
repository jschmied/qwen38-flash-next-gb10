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

