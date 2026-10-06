---
license: apache-2.0
base_model: Aleph-Alpha/Kolibri-1
language:
- en
- de
tags:
- speculative-decoding
- eagle3
- draft-model
- kolibri
datasets:
- mgoin/open-perfectblend-glm5.2-regen
- FreedomIntelligence/sharegpt-deutsch
- mayflowergmbh/alpaca-gpt4_de
---

# Kolibri-1 EAGLE-3 drafter (run 11)

A learned draft model for speculative decoding with [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
(78B MoE, 3.5B active). Kolibri-1 ships no MTP head; this drafter adds one. It is small (90M parameters, one layer),
reads Kolibri's own hidden states, and uses Kolibri's own (frozen) embedding and output head, which are not included
here.

Verification is lossless: the target checks every draft, so replies are identical with and without the drafter
(measured: 0 mismatches, greedy and sampled).

## Results

**End to end in TensorFold**, one NVIDIA DGX Spark (GB10), Kolibri-1 FP8, one stream, 256 tokens a prompt, prompts no
checkpoint trained on, decode speed relative to the engine's copy drafting alone:

| | SWE agent turns | chat | German | mean |
|---|---|---|---|---|
| greedy, depth 1 | 1.23x | 1.30x | 1.23x | **1.25x** |
| greedy, depth 2 | 1.22x | 1.34x | 1.22x | **1.26x** |
| sampled as served, depth 1 | 1.26x | 1.26x | 1.16x | **1.23x** |
| sampled as served, depth 2 | 1.22x | 1.28x | 1.11x | 1.21x |

Copy drafting alone runs at 43-49 tok/s here, the learned drafter at 48-64 tok/s. Four streams together: neutral
(110.0 vs 109.6 tok/s). Recommended depth: 1 (2 pays on chat).

**Acceptance** on Kolibri's own served output that no checkpoint trained on (drafts accepted in a row per round of 3,
32k draft vocabulary, as served):

| | SWE | chat | German |
|---|---|---|---|
| accepted per round | 1.38 | 1.23 | 0.95 |
| step 1 / 2 / 3 agreement | 0.714 / 0.566 / 0.484 | 0.637 / 0.543 / 0.495 | 0.521 / 0.447 / 0.403 |

Verify cost on this hardware is low (a 2-row verify costs 1.15x one decode row, 4 rows 1.39x), which is why short
chains pay. The SWE slice shares repositories with earlier training data, so treat its number as optimistic.

## Files

- `model.safetensors`: drafter weights (linear layers bf16, norms fp32), 90.4M parameters.
- `config.json`: shape (hidden 2560, 20 heads x 128, SwiGLU FFN 4096, 1 layer), the tapped Kolibri layers
  `[44, 47, 49]`, training tokens/steps.
- `draft_vocab.json`: the 32,768 token ids the drafter proposes (94.7 % of Kolibri's generated tokens fall in it).

## Architecture

EAGLE-3 style, autoregressive chain:

- input: Kolibri's normed outputs of layers 44, 47 and 49 concatenated and fused by a linear layer (3 x 2560 -> 2560),
  concatenated with the embedding of the token after the row, then a linear 5120 -> 2560;
- one transformer decoder layer (RMSNorm, RoPE theta 10,000, causal attention over the last 2,048 entries, SwiGLU);
- output: RMSNorm, read by Kolibri's own LM head restricted to `draft_vocab.json`;
- chain: step 1 reads Kolibri's real states; later steps read the drafter's own output.

Differences from EAGLE-3: late taps instead of low/mid/high (a layer probe put the next-token information in layers
44-48); a chain instead of a tree (each extra verify row on a MoE brings its own experts); a small L1 term beside the
token loss.

## Use (TensorFold)

Branch [`jschmied/TensorFold:kolibri1-drafter`](https://github.com/jschmied/TensorFold/tree/edd6aea9079a9c60929b83581bba66734cd7cbf7)
(not merged upstream yet):

```bash
TENSORFOLD_KOLIBRI_DRAFTER=/path/to/Kolibri-1-EAGLE3-drafter \
TENSORFOLD_KOLIBRI_DRAFTER_DEPTH=1 \
tensorfold serve /path/to/Kolibri-1 --name kolibri-1 --parallel 6
```

Copy drafts from the context are tried first; on a miss the learned chain is verified. The weights follow
[model_mt.py](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f93968058a9f834538b5eddc7a02f3ba6c86296d/tools/kolibri/drafter/model_mt.py)
naming, so other engines can port the forward from there.

## Training recipe

All on one DGX Spark. Kolibri-1 runs in TensorFold and prefills each training conversation; the drafter learns from
Kolibri's own states and output distribution at every position (target-distilled; no target weights change).
Scripts: [tools/kolibri/drafter](https://github.com/jschmied/qwen38-flash-next-gb10/tree/f93968058a9f834538b5eddc7a02f3ba6c86296d/tools/kolibri/drafter)
(`train_mt.py` prefill path, `train_rec.py` serving recordings, `eval_rec.py`, `specbench.py`); the full log with every
result and mistake is
[notes/kolibri-drafter-plan.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f93968058a9f834538b5eddc7a02f3ba6c86296d/notes/kolibri-drafter-plan.md).

| stage | start | data | loss | result on unseen data |
|---|---|---|---|---|
| runs 1-3 | scratch, 1 tap (layer 49) | 206 SWE-agent conversations from our earlier agent runs, Kolibri's among them (6.9M tokens), then chat text | L1 to next state + 0.1 CE, 3-step rollouts from run 3 | baseline |
| run 9 | run 3, 3 taps | 45M tokens of GLM-5.2's regeneration of Open-PerfectBlend | same | SWE +14 %, chat +27 % |
| run 10 | run 9 | Kolibri's own served output (recorded while serving, 1.2M generated rows), SWE once / chat twice, lr 1e-4 | top-32 soft CE + L1 | German +14 %, chat +7 % |
| run 11 (this) | run 10 | 60M GLM-5.2 regen tokens + 12.8M German text (packed), 48,245 steps | KL to Kolibri's distribution + 0.1 L1, 32k vocab | chat +5.5 %, German +14 % |

Settings for run 11: AdamW (0.9, 0.95), lr 2e-4 constant after 100 warmup steps, gradient clip 0.5, windows of 2,048
positions, rollout weights 0.8^step, ~2,090 tokens/s, 9.7 h.

What we learned the hard way:

- **Data volume and variety decide.** ~1M rows of Kolibri's own output, repeated, were memorised (no gain on unseen
  data); 45M tokens of broad regenerated text generalised. Never more than 2 passes over a recording.
- **Token-level loss beats state regression.** On the same 10M tokens, KL to the target's distribution gained ~2x what
  EAGLE-1's L1 + CE gained, and trained 17 % faster with the 32k vocabulary.
- **Keep every language in every batch.** A 26M-token batch without German cost 18 % German acceptance.
- **Evaluate on repositories the drafter never saw.** A held-out set from a repository that also appeared in training
  showed +9 % while unseen repositories showed nothing.

## Training data and licences

| source | used for | licence |
|---|---|---|
| [mgoin/open-perfectblend-glm5.2-regen](https://huggingface.co/datasets/mgoin/open-perfectblend-glm5.2-regen) (rev 003f54db, shards 0, 1-9, 18, 26, 28, 29) | text through Kolibri's prefill (105M tokens) | "other" (GLM-5.2 regeneration of [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend), Apache-2.0) |
| [FreedomIntelligence/sharegpt-deutsch](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-deutsch) | German text (4.6M tokens) and German prompts | Apache-2.0 |
| [mayflowergmbh/alpaca-gpt4_de](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de) | German text (8.2M tokens) and German prompts | **none stated**; derived from alpaca-gpt4, whose data is CC BY-NC 4.0 (GPT-4 outputs) |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | prompts Kolibri answered while serving (its own replies were trained on) | MIT, ODC-BY |
| SWE-bench Verified / Multilingual tasks | prompts for Kolibri's agent runs (its own replies were trained on) | MIT |

The weights are released under Apache-2.0. Note on data: part of the German training text (alpaca-gpt4_de) carries no
licence and derives from non-commercial data; check whether that matters for your use.

## Limits

- Measured on one GB10 only, single stream; at four streams the gain is neutral.
- German drafts least well (step-1 agreement ~0.52).
- Needs an engine that exposes Kolibri's layer-44/47/49 states at decode (TensorFold branch above).
