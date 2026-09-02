# MTP acceptance instability — upstream survey (2026-09-02)

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
