# MTP acceptance instability — upstream survey (2026-09-02)

> **Superseded in part the same day.** Rank 1's mechanism (stale state on *resume*) cannot be the
> whole story: `NOCACHE` (no resume at all) still flips (finding 41). The PRs it names do move the
> rate, though (finding 43) — through what they change besides the resume path. Rank 2 (shape-
> bucketed tactic) is dead: identical prompts draw differently within one start (finding 42).

Symptom: MTP n=5 acceptance bimodal per start (9 / 66 / 26 %) and per turn, with a
bit-deterministic target (finding 40). Survey of vllm / flashinfer / sglang threads, ranked by
fit. P = proven in the thread, S = speculated.

## Ranked candidates

| rank | mechanism | explains | evidence | our test |
| --- | --- | --- | --- | --- |
| 1 | **Stale GDN state on every prefix-cache resume on THIS build**: `cache_config.block_size` resolves to 8 while the mamba block is 1600, so `_mamba_block_aligned_split` aligns in units of 8 and every resume continues from a GDN state up to one mamba block stale — silently. Voids the #51113 guarantee. | per-turn flips (every turn of the agent loop is a resume) | vllm#53142 + comment by fabiopili — **same build `0.1.dev20073`, same model, same GPU**; PRs #53798 (site 1, resume >54k IMA = #54173) and #54076 (site 2, split units), both open | `NOCACHE` arm in `mtproot`; then apply #53798 + #54076 |
| 2 | **Launch-to-launch forward-pass variance** from startup kernel/tactic selection (FlashInfer autotune, cuBLAS heuristics) | per-start bias | vllm#54506 correction: two launches of the non-spec engine, identical config, logits differ by up to 5.25 nats on a hybrid+MTP model; #53436 (acceptance↔throughput r=0.98, DSpark n=5, SM120); flashinfer#3537/#3186 (tactic per launch, frozen into the graph) | `--enable-flashinfer-autotune=false` or a pinned `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, 5 starts, compare turn-1 top-8 logprobs across starts (the #54506 method) |
| 3 | **Drafter QSA top-k nondeterminism**: the MTP layer's only attention layer is QSA; the drafter shares the target's `topk_indices_buffer` but runs `persistent_topk` on every draft step (`index_share_for_mtp_iteration` absent from the HF config); tie-set bug vllm#54521 / #51782 | per-turn, part of per-start | P at kernel level; community overlay `VLLM_QSA_EXACT_TOPK` (blazux #3): 0/50 unstable, prefill −8…−40 %, decode unchanged; docai.hu: after it, MTP=2 still not greedy-equivalent | exact top-k on target AND drafter, 3 starts × 8 turns |
| 4 | Widened QSA ring at n=5 (rejection safety unverified) | both | our own caveat (S); spread also seen at legal depths → unlikely primary | `N4` arm in `mtproot` |
| 5 | EAGLE one-block drop / first-repeat cache miss (#53504, #50438) | TTFT per turn only, not acceptance | P | separate TTFT from decode in the turn timing |

## Ruled out from our facts

FULL-cudagraph-only defects #53051 / #49918 / #49010 (we run PIECEWISE); PD #54392; PP #54709;
mode=none/ngram #40738; monotone drift #41838 (fixed ≥0.21); async+spec races #38556 / #45100
(merged, in our build; align path is the sync CPU path anyway). Async scheduling is a weak
candidate: no open thread matches, and #54173 reports `--no-async-scheduling` changes nothing.

## Post-dates our build

vllm#54513 (**merged 2026-09-02**): separate prefill/decode QSA indexer paths for spec decode,
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 128→512. Touches the drafter's indexer — re-measure on a
build that has it before drawing conclusions about the drafter.

## Reference points

SGLang reports accept length 3.3 (B200 TP4) and ~2.3 (2.0–2.7) sustained on DGX Spark with
index reuse in the drafter. Our good regime (4.3) is above that; our bad regime (1.5) is below.

## Field update 2026-09-03

- **#54076 rebased** (mergeable again, semantics unchanged); kamb-code rebuilds #53479 as a
  narrow follow-up on top of #54076 + #54713.
- **#54713** (open, tobymao): under EAGLE/MTP the Mamba retention keeps a state only *at* each
  boundary, one block above what the EAGLE look-up requests, so the Mamba prefix hit is always 0
  (our `prefix_cache_hits_total` counts attention blocks). Verified on a 4× DGX Spark GLM-5.3
  deployment. Queued here as a TTFT/hit-rate test.
- **#54997** (closed by author, no comment): compressed-tensors NVFP4 *drafter* → 0 % acceptance
  under CUDA graphs on B200, eager fine. Same symptom class, different cause (our drafter is
  unquantised and our flip survived eager).
- **#54974** (open): modelopt NVFP4 MoE keeps only the gate global scale for the fused w13 and
  warns when gate ≠ up. **Checked our checkpoint offline** (`qwen38-flash-next-fp8head`, F32
  scalar `weight_scale_2` per expert): gate == up on every pair sampled (28 pairs across 8 layers
  × 4 experts, plus 3000 sequential pairs) — not affected.

## Field note 2026-09-03 — llama.cpp PR #28136 (PLE table direct reads)

llama.cpp keeps the Qwen n-gram/PLE table on the SSD via mmap; a prefill touches ~16 rows per
token (our `ple-access-pattern.md`: ~2.5 KB useful per token, ×26 page amplification). PR #28136
stages the row ids per ubatch, sorts/dedups them and reads with parallel `pread()` workers
(`--lazy-mode on-direct`): DGX Spark cold prefill 300 → 750–800 tok/s, major faults 200k → <100,
decode unchanged, table stays on disk. Open, one approval, ggerganov asked for a cleaner shape.

Relation to our stack: vLLM's PLE offload (`VLLM_PLE_CPU_OFFLOAD=1`, our backport) keeps the
FP8 table RAM-resident (~51 GB of unified memory) and gathers on the CPU, so our prefill is
already ~2400–2600 tok/s (today's TTFT run: 7503 tok in 3.12 s, 29263 in 11.25 s, stock) — about
3× their *after* number. The transferable idea is the other way round: an SSD-backed gather
(sorted/deduped row ids, batched `pread`/io_uring into the connector's staging buffer) would free
~50 GB of unified memory for KV cache / concurrency at the cost of ~6 MB/s of scattered 160-byte
reads at full prefill speed — well inside NVMe IOPS. A feature, not a fix; parked here.
