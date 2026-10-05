# Drafter surveys (2026-10-05)

Two web surveys (subagents, sources linked inline; [R] reported, [M] computed from tensor shapes, [I] inferred).

## MTP heads and draft heads compared with our Kolibri drafter

Our drafter is the right **shape**, but its acceptance sits well below peers at the same depth. At 3 drafts we get an accept length of about 2.13 on SWE (1 + 1.13) and 1.76 on chat. Comparable heads report 2.7–3.6 at depth 3. Sizes and layer counts below are read from each repo's config.json and safetensors headers; nothing was downloaded beyond headers. "AL" means accept length per round, including the bonus token. **[M]** marks numbers I computed from tensor shapes. **[R]** marks numbers the source reports. **[I]** marks my own inference.

### (1) Native MTP heads

| Model (total/active, hidden) | MTP layers | Each MTP layer | Input wiring | Emb / head | MTP params (% of total) | Training, loss weight | Reported acceptance | Src |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3/R1 (671B/37B, 7168) | 1 | full MoE (256e top-8, w2048, +1 shared), MLA | concat(RMSNorm h, RMSNorm emb) → Linear 2d→d | shared | 14B (2.0%) [R] | joint pretraining; λ 0.3 then 0.1 | 2nd token accepted 85–90%, 1.8× TPS at depth 1; R1 AL 2.70 at d=7 [R] | [cfg](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json), [rep](https://arxiv.org/pdf/2412.19437), [NemoS](https://arxiv.org/pdf/2604.12374) |
| DeepSeek-V4-Flash (284B/13B, 4096) | 1 | full MoE (256e top-6, w2048, FP4, +shared) | separate e_proj and h_proj (d→d each) | shared | ~6.5B (2.3%) [M] | joint | DSpark gives +60–85% per-user speed over this MTP-1 [R] | [cfg](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [DSpark](https://arxiv.org/html/2607.05147v1) |
| **DeepSeek-V4.1-Flash, native DSpark** (552B backbone / 8–16B, 5120) | 3, parallel block γ=5 | MoE (128e top-3, w2304, +shared), SWA 128, Markov head rank 256, confidence head | **Linear 3d→d over target layers 37/38/39** | shared | ~14B (~1.9%) [M] | trained on target hidden states (CE + distribution + confidence) | as above [R] | [cfg/card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Qwen3-Next-80B-A3B (80B/3B, 2048) | 1 | full MoE (512e top-10, w512, +shared), gated attention | pre_fc_norm(h), pre_fc_norm(emb) → fc 2d→d | shared | 1.65B (2.0%) [M] | joint, multi-step | AL 3.33 at d=7, SPEED-Bench [R] | [card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [NemoS](https://arxiv.org/pdf/2604.12374) |
| Qwen3.5/3.6-35B-A3B (36B/3B, 2048) | 1 | full MoE (256e top-8, w512, +shared) | same as Qwen3-Next | shared | 0.845B (2.35%) [M] | joint | **AL 3.21–3.58 at 3 steps**, 4.49–5.61 at 7, c=1 [R] | [z-lab card](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3.8-Flash-Next (125B + 51B n-gram / 6B, **2560**) | 1 | full MoE (512e top-10, w640, +shared), indexer attention, hyper-connections | separate fc_hidden and fc_emb (d→d each) | shared | 2.61B by tensors [M]; card says "4B MTP" (1.5–2.2%) | "trained with multi-steps" | none stated | [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| GLM-4.5-Air (106B/12B, 4096) | 1 | full MoE (128e top-8, w1408) | eh_proj 2d→d | stored copies | ~2.4B unique (2.2%) [M] | joint | none stated | [cfg](https://huggingface.co/zai-org/GLM-4.5-Air) |
| GLM-4.7-Flash (31B/~3B, 2048) | 1 | MoE (64e top-4, w1536), MLA | eh_proj | stored copies | ~0.64B unique (2.1%) [M] | joint | none stated | [cfg](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| GLM-5 (744B/40B, 6144) | 1 | MoE (256e, w2048) | eh_proj | – | included in total | **3 MTP steps trained with shared params** | AL 2.76 vs DSV3.2's 2.55 at 4 steps (private set) [R] | [rep](https://arxiv.org/pdf/2602.15763) |
| MiMo-7B (7.8B dense, 4096) | 1 | dense FFN 11008 | input_proj 2d→d | shared | 0.21B (2.7%) [M] | pretraining + SFT, frozen during RL | ~90% acceptance at depth 1 [R] | [card](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) |
| MiMo-V2-Flash (309B/15B, 4096) | 3 | **dense FFN 16384, SWA** | eh_proj 2d→d | shared | 0.33B per layer, 0.99B total (0.32%) [M/R] | 1 head in pretraining (λ 0.3 then 0.1), copied ×3 in post-training | **AL 2.8–3.6 at depth 3**, 1.8–2.7× [R] | [rep](https://arxiv.org/pdf/2601.02780) |
| LongCat-Flash (560B, 6144) | 1 | dense FFN 12288, MLA | eh_proj | own embedding copy | 0.39B excl. embedding (0.07%) [M] | added mid-training | >90% acceptance [R] | [rep](https://arxiv.org/html/2509.01322v1) |
| LongCat-Flash-Lite (68.5B / 2.9–4.5B, 3072) | 1 | dense FFN 6144, MLA | eh_proj | own embedding copy | **~0.11B excl. embedding (0.16%)** [M] | late pretraining | ~90% (secondary source only) | [cfg](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite) |
| Step-3.5-Flash (196B/11B, 4096) | 3 | dense FFN 11264, SWA, gated attention | eh_proj | each layer stores its own head (528M) | ~0.84B excl. heads (0.42%) [M] | module 1 trained in pretraining; 2–3 cloned, then fine-tuned | 100–350 tok/s; no AL given | [card](https://huggingface.co/stepfun-ai/Step-3.5-Flash) |
| Nemotron-3-Super (124B/12B, 4096) | 2, shared weights | attention layer + LatentMoE layer (512e) | eh_proj | shared | 2.94B (2.4%) [M] | joint, λ 0.3, continued in SFT, final "MTP healing" stage | AL 3.45 at d=7 [R] | [rep](https://arxiv.org/pdf/2604.12374) |
| Ling-flash-2.0 / 2.6-flash (103–107B / 6.1B, 4096) | 1 | full MoE (256e, w1024), MLA | eh_proj | – | 3.3B (3.1%) [M] | joint, λ 0.1; the 2.0 chat release ships no MTP | none stated | [rep](https://arxiv.org/html/2510.22115v2) |
| MiniMax-M2 / M2.5 | config declares 3×1 | – | – | – | **MTP weights not released** | 1 module in pretraining, expanded to 3 during decay | – | [issue](https://github.com/MiniMax-AI/MiniMax-M2/issues/47) |
| Kimi K2 / K2.5 / K3, Nemotron-3-Nano, gpt-oss | **none** (num_nextn = 0 or field absent) | – | – | – | – | Kimi and gpt-oss have no official drafter except NVIDIA's gpt-oss EAGLE-3 (table 2) | – | configs |

### (2) Third-party drafters

| Drafter | Layers × hidden / FFN | Taps | Depth | Params | Reported AL | Src |
|---|---|---|---|---|---|---|
| EAGLE-3 Qwen3-30B-A3B (lmsys) | 1 × 2048 / 12288, 32k draft vocab | 3 | 3 steps | 183M | only published as images | [card](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex) |
| EAGLE-3 GLM-4.7-Flash (thoughtworks) | 1 × 2048 / 8192, 32k vocab | 3 | 3 steps, 6 tokens (tree) | 145M | SWE-V 3.10, MT-Bench 2.88, mean 3.29; 1.66× at B=1 | [card](https://huggingface.co/thoughtworks/GLM-4.7-Flash-Eagle3) |
| EAGLE-3 gpt-oss-120b (NVIDIA v3) | 1 × 2880 / 16384, full vocab | 24/30/36 (late) | d=7 | 787M | 2.95 average (coding 3.28) | [card](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-v3) |
| EAGLE-3 Llama-4-Scout (lmsys) | 1 × 5120 / 32768 | 3 | 3 steps | 846M | only published as images | [card](https://huggingface.co/lmsys/SGLang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-SpecForge) |
| EAGLE-3 Kimi-K2.5 (lightseek) | 1 × 7168 / 12288 | 3 | 3 steps, chain | 3.18B | MT-Bench 2.69, HumanEval 3.29 | [card](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) |
| DFlash Qwen3-Coder-30B-A3B | 8 × 2048 / 6144 | 5 | block 16 | 474M | 6.4–8.1 on code; 2.6–3.5× | [card](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| DFlash Qwen3.5/3.6-35B-A3B | 6 (5 SWA + 1 full) × 2048 / 6144 | 8 | block 4 / 8 / 16 | 386M | 3.14–3.60 / 4.43–5.90 / 5.34–8.22 | [card](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| DFlash gpt-oss-120b | 8 × 2880 / 7680 | 5 | block 4 / 10 | 785M | 2.7–3.3 / 3.7–5.4 | [card](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| DFlash MiniMax-M2.5 | 5 × 3072 / 6144 | 6 | block 8 | 1.79B | – | [cfg](https://huggingface.co/z-lab/MiniMax-M2.5-DFlash) |
| DSpark Kimi-K3 (Inferact) | 5 dense × 7168 / 14336 | 5 | block 7 | 3.56B | mean 3.85; SWE-bench Pro 3.35; MT-Bench 3.14 | [card](https://huggingface.co/Inferact/Kimi-K3-DSpark) |
| NeMo AutoModel DFlash recipe | defaults: 5 layers, block 16, taps spread evenly, loss decay γ = 7/5/4 for block 16/10/8 | | | | none given | [docs](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding) |

The DSpark paper runs EAGLE-3 with 1 layer and DSpark/DFlash with 5 layers. It reports DSpark's accept length +30.9% over EAGLE-3 on Qwen3-4B ([paper](https://arxiv.org/html/2607.05147v1)).

### (3) Takeaways for our drafter

- **One layer is typical.** It is the norm for every native MTP head except MiMo-V2 and Step-3.5 (3 separate layers), Nemotron (2 with shared weights) and DeepSeek-V4.1's DSpark (3). Every EAGLE-3 drafter in the survey is 1 layer. Only the parallel drafters (DFlash, DSpark) go to 5–8 layers. [R]
- **Dense vs MoE:** labs that care most about decode cost pick dense (MiMo-V2, Step-3.5, LongCat); Qwen, GLM, DeepSeek and Ling reuse a full MoE block. For Kolibri, a MoE MTP layer would store about 1.5B params but read only about 28M FFN weights per token (7×3×2560×512). That is about the same as our 31.5M dense FFN, so MoE buys capacity at roughly the same per-token cost, paid in memory. [M/I]
- **Parameter share:** our 90M is 0.12% of total and about 2.6% of active. MoE MTP heads are 2–3% of total. Dense heads are 0.07–0.42% of total but 2.2–3.6% of active per layer. LongCat-Flash-Lite (~0.11B, 1 dense layer) is the closest analog in size. **By active share we are normal, not oversized.** [M]
- **FFN width is our one real outlier.** EAGLE-3 drafters for MoE targets use an FFN of 4–6.4× hidden (GLM-4.7-Flash 8192/2048, Qwen3-30B 12288/2048, gpt-oss 16384/2880, Scout 32768/5120). The dense MTP heads use 2.75–4× (MiMo-V2 16384/4096, Step 11264/4096). Ours is 4096/2560 = 1.6×. A 10240-wide FFN would add about 47M params. [M/I]
- **Input wiring:** native MTP takes only the last hidden state plus the embedding (concat 2d→d). That works because the target is trained jointly with it. Post-hoc drafters use several taps: EAGLE-3 uses 3, DFlash/DSpark 5–8. DeepSeek's own post-hoc DSpark taps the last 3 layers through a Linear 3d→d, which is exactly our 44/47/49 design. NVIDIA's gpt-oss EAGLE-3 also uses late taps (24/30/36). Our wiring has precedent. [R]
- **Multi-step training:** GLM-5, Nemotron and Qwen train shared parameters over 2–3 steps specifically to lift later draft positions. Our 3-step rollouts match that practice. [R]
- **Acceptance is the gap.** Depth-3 peers report AL 2.8–3.6 on code/math and 2.7–3.2 on chat (MiMo-V2, Qwen3.5-35B MTP, the Kimi EAGLE-3). The closest 1-layer peer, the GLM-4.7-Flash EAGLE-3 at about 145M params, reaches 3.10 on SWE-Verified, though with a 6-token tree. We are at 2.13 on SWE and 1.76 on chat, which is about 0.8–1.4 tokens per round short. [R/M]
- **Undersized or under-trained?** Similar-sized 1-layer drafters reach about 3. That points first at training data and on-policy regeneration, then at FFN width, rather than layer count. Benchmarks and sampling settings differ across all these sources, so the size of the gap is only approximate. [I]
- **What more depth or layers would buy:** going past depth 3 to 5–7 pays off mainly with parallel block drafters (DSpark, DFlash at 5 layers, 0.4–3.6B params). A 1-layer autoregressive head flattens out around AL 3.3–3.5 even at d=7 (Qwen3-Next 3.33, Nemotron 3.45, gpt-oss EAGLE-3 2.95). [R]

---

# Drafter training stories

I covered 15 drafter-training stories below, each with its source. Everything is **reported** by the source unless marked *[inferred]*. I recall two older facts from memory and did not re-check them this session; those are marked *[unverified]*. Several summaries were silent on compute and hyperparameters, so those cells read "n/s" (not stated).

## (1) Stories

| Who | Target | Drafter | Data (size, regenerated?) | Compute | Acceptance / speedup |
|---|---|---|---|---|---|
| [EAGLE-3 paper](https://arxiv.org/html/2503.01840) | Llama-3.1-8B, R1-Distill-8B | 1 decoder layer, 3-tap fusion, training-time test (the training loop feeds back the drafter's own predictions) | ShareGPT 68K + UltraChat 464K (+OpenThoughts-math for R1), **target-regenerated**; AdamW lr 5e-5, clip 0.5 | n/s | acceptance length 6.13 on MT-bench (EAGLE-2: 4.05); 6.93 on GSM8K for R1; SGLang throughput **1.38x at batch 64** |
| [SpecBundle / SpecForge](https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/), [paper](https://arxiv.org/html/2603.18567v1) | Qwen3-235B-A22B, Coder-480B, Llama-4, Kimi-K2, Ling | 0.6B EAGLE-3 | Open-PerfectBlend **1.4M, regenerated**, 2 epochs ([235B card](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-235B-A22B-Instruct-2507-SpecForge-Meituan)) | 235B: 8 GPUs, 1.62 s/step | up to 4.48x on LiveCodeBench; 1.35x over older checkpoints |
| [LMSYS SpecForge v1](https://lmsys.org/blog/2025-07-25-spec-forge/) | Llama-4 Scout / Maverick (MoE) | EAGLE-3 | ShareGPT + UltraChat, 320K | n/s | 2.0x Scout, 2.18x Maverick (MT-bench) |
| [Meta, Llama at scale](https://arxiv.org/html/2508.08192) | Llama-3.1/3.3, Llama-4 Scout/Maverick | **3-layer dense** EAGLE (10x fewer params than an MoE drafter) | the target's own SFT set, 48k iters × 2M tokens (~96B tokens) *[inferred product]*; Adam lr 2e-4, wd 0.1; loss L1(hidden) + 0.1·CE | n/s | tokens per verify call 2.75–2.94; ~4 ms/token at batch 1 on 8×H100; 1.4–2.0x at large batch |
| [Red Hat data study](https://developers.redhat.com/articles/2026/07/06/smarter-data-generation-faster-speculator-training) | Qwen3-30B-A3B, gpt-oss-20b/120b, Qwen3-32B, Laguna | EAGLE-3 | 500K prompts (Magpie 300K + UltraChat 200K), self- vs cross-distilled | 8×A100/H100 | cross-distilled from a bigger same-family model: 30B-A3B 2.77→**3.05**; gpt-oss-20b 2.17→2.30; cross-family *loses* on gpt-oss-120b (2.65 self vs 2.40) |
| [Red Hat gpt-oss-120b](https://x.com/RedHat_AI/status/2039053334145888277) | gpt-oss-120b (MoE) | 0.9B EAGLE-3 | n/s | n/s | acceptance length at k=5: math 3.29, HumanEval 3.01, translation **2.52** |
| [NVIDIA ModelOpt](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-throughput) | gpt-oss-120b | 0.8B EAGLE-3 | 1.6M, prompts only, **regenerated by the target** | 4.8e20 FLOP, ~2500 kWh | 1-token drafts, acceptance length 1.72–1.84; tuned for high concurrency |
| [z-lab DFlash](https://arxiv.org/html/2602.06036) | Qwen3-4B/8B, Coder-30B-A3B, gpt-oss | 5-layer (8 for Coder) block diffusion, 16-token blocks, target features injected into every layer's KV cache | ~800K Nemotron-v2 + CodeAlpaca, **regenerated**; lr 6e-4, 6 epochs, 3072-token sequences, exp-decay position weights | n/s | Qwen3-8B 4.9x (2.4x over EAGLE-3); **c=1 5.1x → c=32 2.8x**; Coder-30B MoE 3.2x at c=8; 289K samples beat EAGLE-3 trained on 1.4M |
| [AMD (vLLM blog)](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct) | MiniMax-M3 MXFP4 (420B MoE), Kimi-K2.5 | EAGLE-3, **FP8 drafter** | on-policy, chat + raw-completion endpoints | n/s | acceptance length 2.80 (code 3.32, roleplay 2.01), flat from 1K to 32K context; FP8 drafter 1.76–2.00x vs BF16 1.69–1.90x |
| [Nebius MTP fine-tune](https://huggingface.co/nebius/MTP-DeepSeek-V3-0324) | DeepSeek-V3-0324 | its native MTP head, re-trained | 660K Infinity-Instruct with **V3-regenerated** answers, 1 epoch, LK loss (targets acceptance directly), K=6 | n/s | acceptance length at K=7: **3.20→4.83** (temp 0), 3.09→4.68 (temp 1) |
| [DeepSeek-V3 report](https://arxiv.org/html/2412.19437v1) | DeepSeek-V3 | native MTP | pretraining only | — | 2nd-token acceptance 85–90%, 1.8x TPS |
| [Snowflake Arctic](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) | Llama-class | MLP / LSTM speculator, ~2B | single-stage training on target-generated UltraChat + MagiCoder | n/s | 3.1x higher acceptance than two-stage training; 2.45x on ShareGPT |
| [IBM](https://arxiv.org/html/2404.19124v1) | Llama-3, Granite | multi-head MLP | stage 1 on raw text at 4k context, stage 2 on target-generated 256-token samples | n/s | 2-stage recipe |
| [FastDraft (Intel)](https://arxiv.org/abs/2411.11055) | Phi-3-mini, Llama-3.1-8B | small standalone drafter | ~10B pretraining tokens, then target-synthetic alignment | 8× Gaudi2, <24 h | up to 3x on code, 2x on other tasks |
| [Baseten live](https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/) / [Together ATLAS](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas) | Kimi-K2, DeepSeek-V3.1 (MoE) | EAGLE-3 trained online; static + adaptive pair | live serving hidden states | n/s | Baseten: median acceptance **+20%**, +100% on narrow traffic; ATLAS: up to 500 TPS (3.18x) on V3.1 |

## (2) Mistakes and lessons

- **A frozen pretraining head misses chat tokens.** DeepSeek's MTP gets ~83% at k=1 on ShareGPT vs ~90% in the paper. Template special tokens and the RLHF style shift are what it misses ([ROCm/ATOM#609](https://github.com/ROCm/ATOM/issues/609)).
- **Updating the target without the drafter costs acceptance.** Fine-tuning or RL on the base model degrades the drafter unless it is retrained alongside ([Baseten](https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/)).
- **Regenerating with the target helps, but only modestly in one test.** SpecForge measured +5.3% average throughput, overturning the claim that EAGLE is insensitive to its data ([SpecForge](https://arxiv.org/html/2603.18567v1)).
- **A stronger teacher is not always better.** Same-family cross-distillation helps (+10%), but cross-family lost on gpt-oss-120b (2.40 vs 2.65). Adding ~50K self-distilled samples to cross data lifted it to 2.60 ([Red Hat](https://developers.redhat.com/articles/2026/07/06/smarter-data-generation-faster-speculator-training)).
- **Overtraining:** 3–4 epochs is the optimum, and two-thirds of the gain lands in the first two (Red Hat, same post).
- **MoE drafters lose.** At equal FLOPs, MoE drafts reached 1.61–1.79 vs 2.55–3.48 for dense ([SpecForge](https://arxiv.org/html/2603.18567v1)). Meta also kept dense FFNs.
- **TTT length depends on the task.** MT-bench peaks at 3 steps, while GSM8K and Math500 keep improving to about 13, at proportional training cost ([SpecForge](https://arxiv.org/html/2603.18567v1)).
- **Thinking text is harder to draft.** Qwen3 with thinking on loses ~17% throughput ([HF blog](https://huggingface.co/blog/lujangusface/tw-eagle3-gpu)).
- **Creative text and translation lag.** Translation reaches 2.52 vs math 3.29 (Red Hat 120b). Roleplay reaches 2.01 vs code 3.32 (AMD).
- **English-only data gives no multilingual numbers.** NVIDIA could not quantify acceptance on Dutch ([HF discussion](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-throughput/discussions/2)). A dedicated pretrain-then-finetune per language works ([Yi et al.](https://arxiv.org/abs/2406.16758)). Osprey reports its largest gains on out-of-domain and multilingual data ([Osprey](https://arxiv.org/abs/2609.09338)).
- **Tiny data looks broken.** An EAGLE-3 trained on 5K ShareGPT samples underperforms MTP ([speculators#751](https://github.com/vllm-project/speculators/issues/751)); the issue has no diagnosis yet.
- **Single-layer input fusion and few taps underperform.** Injecting target features into every layer's KV cache beats single-layer fusion, and 5 target layers beat 3 ([DFlash](https://arxiv.org/html/2602.06036)).
- **A quantized target was fine for AMD.** An MXFP4 target with an FP8 drafter showed no acceptance loss (AMD).
- **Domain shift is large.** Generic drafters drop sharply on domain-specific targets. Synthetic data recovers 80–93% of what real user queries give ([2503.07807](https://arxiv.org/abs/2503.07807)).

## (3) What made the speedups real

- **CUDA graphs, greedy verify, sharded top-k before all-gather, FP8:** drafter latency fell 1.47 → 0.47 ms/token, reaching 91% of the theoretical speedup (Snowflake).
- **Meta:** tree attention split into prefix and suffix, compiled sampling (1.5x), and fewer CPU-GPU syncs (94% GPU use, 8–12% better inter-token time). This took speedup from collapsing to 0.7x at batch 48 to 1.4–2.0x at scale.
- **Plain chains work.** A 3-token chain with no tree still gives 1.38x at batch 64 (EAGLE-3). NVIDIA ships a 1-token drafter for high concurrency.
- **Reduced draft vocabulary:** 10–32k tokens is the default ([speculators v0.3](https://vllm.ai/blog/2025-12-13-speculators-v030)).
- **Gains shrink with concurrency:** DFlash goes from 5.1x at c=1 to 2.8x at c=32. AMD sees the largest gains at low concurrency.

## (4) Data-scaling evidence

- EAGLE-3: acceptance keeps rising with 8x more data, where EAGLE-1/2 had plateaued.
- [Scylla](https://arxiv.org/html/2505.07858): acceptance rises log-linearly with pretraining tokens (R²=0.89).
- Together, customer-traffic fine-tunes: **20M tokens gave >1.10x and 50M tokens >1.20x** over their base speculator ([Together](https://www.together.ai/blog/customized-speculative-decoding)).
- Red Hat: gains flatten after epoch 3, so adding data beats adding epochs *[inferred]*.
- DFlash's 289K samples beat EAGLE-3's 1.4M: architecture can outweigh 5x more data.
- Online serving-distribution distillation: acceptance +0.1 to +0.65, latency 1.42–2.17x lower ([OSD](https://arxiv.org/abs/2310.07177)).
- Meta trained on roughly 96B tokens, about 2000x our 45M *[inferred]*. We are far from saturation.

## (5) Recommendations for Kolibri-1

1. **Keep the recipe, add a small self-data mix.** Your GLM-regenerated data run through the target's prefill matches the cross-distillation pattern. Red Hat's hybrid gain (2.40→2.60) suggests adding ~50K rows of Kolibri's own outputs *[inferred transfer]*. The memorised 1M self rows are consistent with their 3–4 epoch ceiling, so cap epochs rather than rows.
2. **Scale the cross-distilled data next.** At 45M tokens you are near the bottom of every scaling curve. Together's 20→50M step was worth about +10 points of speedup. Try 100–200M tokens before changing the architecture.
3. **Try 5 taps and per-layer KV injection** before adding drafter depth (DFlash ablation). Keep the FFN dense.
4. **Try longer rollouts for code.** Train with 5–7 TTT steps for SWE traffic, and keep 3 for chat if memory is tight (SpecForge).
5. **Try LK loss** for the fine-tuning on serving recordings. It gave +51% acceptance on DeepSeek MTP, and your German result shows serving data carries most of the domain gain.
6. **Make it continual:** keep a frozen general drafter and periodically fine-tune an adaptive copy on new recordings (ATLAS/Baseten pattern). Retrain whenever Kolibri's weights or template change.
7. **Evaluate per language and domain at the concurrency you deploy.** Report German, English, code and roleplay separately at c=1 and c≥8, because translation and creative text trail code by 0.8–1.3 accepted tokens.
8. **Keep the template exact.** Generate training data with the exact serving chat template, think tokens and special tokens, since template boundaries are where frozen heads lose acceptance.
