POSTED 2026-09-27 (user's go: "post"). vllm-project/vllm#58578 (FR-Spec draft vocabulary for MTP), second data point.

A second data point, on Qwen3.8-Flash-Next (qwen4_exp; the MTP head shares the target's FP8 lm_head, 248,320 × 2,560), one DGX Spark (GB10, sm_121), V2 model runner:

- **32k-id list, frequency-ranked on our own agent output** (99.6 % held-out coverage): **+6.4 % one-stream decode**, +3 % at 4 streams, acceptance unchanged; 3 server starts, 20 of 24 paired wins. 8k and 16k lists gave the same speed but −1.9 / −1.1 pp acceptance ([finding 135](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/notes/determinism-investigation.md#L1521-L1545)).
- **The slice stored as NVFP4** (45 MiB instead of 160 MiB BF16): another −3.4 % per token at one stream, acceptance unchanged, output identical ([finding 234](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/notes/prefill-investigation.md#L3886)).
- **Probabilistic drafting stays exact over a slice** if the slice's logits are scattered into a −inf full-vocabulary buffer, so q = 0 outside it: −5.5 % on sampled code at K=5 ([§5n](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/notes/speed-of-light.md#L1219)).
- Against a larger content-built list (TensorFold's 79,591 ids), our 32k list covers 0.1–0.6 pp less of real text; not worth the 2.4× head bytes ([§5k](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/notes/speed-of-light.md#L1153-L1168)).

Our hook sits on the V2 speculator's local-argmax path ([`dv_patch.py`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/tools/draft_vocab/dv_patch.py), list builder [`build_vocab.py`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/afe0029dbb85fb3028083ac496ca999d5aba8ff3/tools/draft_vocab/build_vocab.py)). Happy to test a PR on GB10.

Written with AI assistance (Claude Code).
