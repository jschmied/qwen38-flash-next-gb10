---
license: apache-2.0
base_model: Aleph-Alpha/Kolibri-1
language:
- en
- de
tags:
- speculative-decoding
- block-drafter
- draft-model
- kolibri
datasets:
- mgoin/open-perfectblend-glm5.2-regen
- FreedomIntelligence/sharegpt-deutsch
- mayflowergmbh/alpaca-gpt4_de
- HuggingFaceH4/ultrachat_200k
---

# Kolibri-1 block drafter

A learned draft model for speculative decoding with [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
(78B MoE, 3.5B active). Kolibri-1 ships no MTP head. This drafter proposes a block of up to 4 tokens per round from
Kolibri's own hidden states. It uses Kolibri's frozen embedding and output head, which are not included here. It
replaces our earlier EAGLE-3 chain drafter (run 11, not published), which the tables compare against.

Verification is lossless: the target checks every draft, so replies are the same with and without the drafter
(measured: 0 mismatches, greedy and sampled).

## Results

**End to end in TensorFold.** Setup:

- one NVIDIA DGX Spark (GB10), Kolibri-1 FP8;
- one stream, 256 tokens per prompt;
- prompts no checkpoint trained on: 10 SWE, 8 chat, 8 German;
- the drafter at its release defaults: depth 2, FP8 weights, NVFP4 head slice.

Decode speed in tok/s:

| | SWE agent turns | chat | German | 4 streams (chat + German) |
|---|---|---|---|---|
| copy drafting only | 46.6 | 48.1 | 50.6 | 113.9 |
| run 11 chain drafter | 57.7 | 61.5 | 66.5 | 115.1 |
| **this drafter, greedy** | **64.3 (1.38x)** | **61.6 (1.28x)** | **69.3 (1.37x)** | **125.5 (1.10x)** |
| copy only, sampled as served | 42.6 | 47.9 | 47.3 | |
| **this drafter, sampled as served** | **62.6 (1.47x)** | **60.2 (1.26x)** | **58.3 (1.23x)** | |

Compared with run 11:

- greedy: +11 % on SWE, chat level, +4 % on German;
- sampled as served: +17 % on SWE, chat −1 %, German +2 %;
- 4 streams: +9 % (run 11 was neutral there, and an earlier bf16 build of this drafter was −17 %).

**Acceptance** on Kolibri's own served output that no checkpoint trained on, using the 64k draft vocabulary:

| | SWE | chat | German |
|---|---|---|---|
| accepted per round of 3 drafts | 1.908 | 1.288 | 1.105 |
| accepted per round of 4 drafts | 2.229 | 1.395 | 1.192 |
| run 11, round of 3 | 1.38 | 1.23 | 0.95 |
| position 1 / 2 / 3 / 4 agreement | 0.852 / 0.650 / 0.497 / 0.392 | 0.696 / 0.445 / 0.296 / 0.196 | 0.622 / 0.383 / 0.249 / 0.168 |

These are development slices: we selected checkpoints on them, and no separate test pool was scored. The SWE slice
shares repositories with the training recordings, so treat its number as optimistic. The end-to-end prompts above
were never used for selection.

## Files

| file | contents |
|---|---|
| `model.safetensors` | drafter weights, bf16; 391.3M parameters. FP8 and NVFP4 are made at load time. |
| `config.json` | shape (see Architecture); the tapped Kolibri layers `[44, 47, 49]`; serving defaults `{"depth": 2, "quant": "fp8", "head": "nvfp4"}`; training steps |
| `draft_vocab.json` | the 60,614 token ids the drafter proposes: every token Kolibri produced in our 6.03M recorded rows |

## Architecture

The block design follows [DFlash](https://arxiv.org/abs/2602.06036): one pass drafts several positions. Three
additions are taken from other work:

- **Input.** Kolibri's normed outputs of layers 44, 47 and 49 are concatenated and fused by a linear layer
  (3 × 2560 → 2560). Context rows follow the EAGLE-3 form: fused state plus the embedding of the next token, then a
  linear layer 5120 → 2560.
- **Body.** 4 transformer layers: hidden 2560, 20 heads × 128, SwiGLU FFN 6144, RoPE theta 10,000. Each layer
  cross-attends to the last 2,048 context rows; the block is causal inside itself.
- **Position 1 is an EAGLE-3 chain head.** Block row 0 is the anchor's own row and sees exactly what a chain drafter
  sees. Its layers started from run 11 and were later trained at 0.1 × the learning rate.
- **Positions 2–4** read a learned mask row plus the anchor's fused state.
- **Predecessor head** ([DSpark](https://arxiv.org/abs/2607.05147)-style, rank 256). Position j ≥ 2 adds
  `U silu(A h + B embed(token before))`, so each row knows its predecessor token. During training the predecessor is
  Kolibri's argmax; at serving it is the drafter's own previous draft.
- **Spine** ([DSpine](https://arxiv.org/abs/2609.36173)-style, rank 256). Before layers 2–4, row j reads row j−1's
  state through RMSNorm → A → silu → U. It is zero-initialised and leaves row 0 untouched.
- **Output.** RMSNorm, then Kolibri's own LM head restricted to `draft_vocab.json`.

**Serving.** Several drafter rounds in a row (depth) can run before the target verifies. Every stream in a decode
round is drafted in one batched pass. On the GB10 the drafter's cost is the bytes it reads per round, not kernel
launches. That is why the body runs as block-FP8 (e4m3, fp32 scale per row and 64 inputs) and the 60k-row head slice
as NVFP4 (ModelOpt recipe): the NVFP4 head keeps the same drafts (960 vs 962 kept on chat) at a quarter of the bytes.

## Use (TensorFold)

The drafter needs a fork branch, not merged upstream:
[`jschmied/TensorFold:kolibri1-drafter`](https://github.com/jschmied/TensorFold/tree/8ea05aae1a89228fb3336d90cf103ca68dfac2f1)
at commit `8ea05aa`.

```bash
TENSORFOLD_KOLIBRI_DRAFTER=/path/to/Kolibri-1-block-drafter \
tensorfold serve /path/to/Kolibri-1 --name kolibri-1 --parallel 6
```

The engine sees a block drafter from `config.json` and uses its serving defaults. Copy drafts from the context are
tried first; on a miss the learned block is verified. Overrides:

| variable | effect |
|---|---|
| `TENSORFOLD_KOLIBRI_DRAFTER_DEPTH` | how many rounds are drafted before the target verifies |
| `TENSORFOLD_KOLIBRI_BLOCK_QUANT` | body weights: `bf16`, `fp8` or `nvfp4` |
| `TENSORFOLD_KOLIBRI_BLOCK_HEAD` | head slice: `bf16`, `fp8` or `nvfp4` |

The fork's GPU test checks that drafted replies equal serial replies in bf16, FP8, and FP8 with an NVFP4 head, for
three streams in one pass, greedy and sampled. Weight names follow
[model_block.py](https://github.com/jschmied/qwen38-flash-next-gb10/blob/4259630bae9135ac863661aad088774e79552a11/tools/kolibri/drafter/model_block.py),
so other engines can port the forward pass from there.

## Training recipe

All training ran on one DGX Spark. Kolibri-1 runs in TensorFold and prefills or serves the training text. The
drafter learns from Kolibri's own states and output distribution at every position (target-distilled); no target
weights change. Text prefills and serving recordings alternate with training, because two 77 GB targets do not fit
in 128 GB.

Scripts are in
[tools/kolibri/drafter](https://github.com/jschmied/qwen38-flash-next-gb10/tree/4259630bae9135ac863661aad088774e79552a11/tools/kolibri/drafter):
`train_block.py`, `model_block.py`, `eval_block.py`, `export_block.py`, `specbench.py`. Every result and mistake is
logged in
[notes/kolibri-drafter-plan.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/4259630bae9135ac863661aad088774e79552a11/notes/kolibri-drafter-plan.md).

| stage | change | dev score after (SWE / chat / German, round of 3) |
|---|---|---|
| run 15 | 4-layer block built on run 11 (fuse, fc, layer 0 copied); position 1 frozen; GLM-5.2 text | 1.30 / 1.04 / 0.82 |
| run 16b–d | + predecessor head; + Kolibri's own served output (recording batches 2–3) | 1.49 / 1.11 / 0.87 |
| run 16u/f | position 1 unfrozen at 0.1 × learning rate, loss weight 2 | 1.62 / 1.14 / 0.92 |
| run 16g/h | + spine (rank 256), all recordings | 1.73 / 1.22 / 0.99 (32k vocab) |
| run 16i (this) | 64k draft vocabulary; German recording batch 4; 45,000 steps | 1.91 / 1.29 / 1.11 |

Settings for run 16i:

- AdamW (0.9, 0.95), learning rate 2e-4 cosine to 2e-5, gradient clip 0.5;
- windows of 2,048 positions; position weights 0.8^j;
- loss: KL to Kolibri's distribution on the draft vocabulary;
- batches are 40 % recordings (mix SWE 0.3 / chat 0.4 / German 0.3) and 60 % GLM-5.2 text.

`training_tokens` in `config.json` is the position in the text stream across runs 15–16i; recording windows are not
counted in it.

What mattered, in order:

- **Draft vocabulary.** Widening the slice from 32k to 64k raised the cap on every class (German most) and needed no
  retraining to score.
- **Unfreezing position 1 at a low learning rate.** This was the largest single step (+0.11 on SWE in 1,621 steps).
  At the full learning rate (run 15) it had forgotten its chain skill.
- **Kolibri's own served output**, mixed in at a fixed share per batch and never repeated more than twice.
- **The spine and the data-token predecessor were neutral within ±0.002.** The spine is kept because it costs
  little and did not hurt.

## Training data and licences

| source | used for | licence |
|---|---|---|
| [mgoin/open-perfectblend-glm5.2-regen](https://huggingface.co/datasets/mgoin/open-perfectblend-glm5.2-regen) (rev 003f54db) | text through Kolibri's prefill | "other": GLM-5.2 (MIT) answers to [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) prompts (Apache-2.0) |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | prompts Kolibri answered while serving; its own replies were trained on | MIT |
| [FreedomIntelligence/sharegpt-deutsch](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-deutsch) | German prompts, plus German text as context | Apache-2.0 |
| [mayflowergmbh/alpaca-gpt4_de](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de) | German prompts, plus German text as context | no tag; a reformat of FreedomIntelligence/alpaca-gpt4-deutsch (Apache-2.0), a German translation of GPT-4 alpaca answers |
| [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | prompts (inherited from run 11) | ODC-BY |
| SWE-bench Verified / Multilingual tasks | prompts for Kolibri's agent runs; its own replies were trained on | MIT |

Only the first user turn of each chat prompt was sent; every recorded answer is Kolibri's own. Third-party answers
(ChatGPT, translated GPT-4) appear only as prefill context, and the training target there is Kolibri's
distribution. The weights are released under Apache-2.0. One caveat: part of the German context text derives from
GPT-4 outputs, so check whether that matters for your use. This is not legal advice.

## Limits

- Measured on one GB10 only. Single stream, plus one 4-stream cell.
- German drafts least well: position 1 agreement is 0.62, against 0.85 on SWE.
- Needs an engine that exposes Kolibri's layer 44/47/49 states at decode (the TensorFold branch above).
- No separate held-out test pool was scored; the acceptance numbers are development slices.
