POSTED 2026-10-02. TensorFold PR #252 comment (2026-10-02).
Confirmed on GB10, Flash Next EXL3 3.05 bpw, this branch. Setup: the table unlocked (on 3.05 the engine mlocks it at startup, so this path only acts when the table does not fit), a 24,576-token prompt, mappings dropped plus page cache evicted, two runs per cell, a new prompt per run, identical tokens in every arm.

| | cold prefill | major faults | table resident after |
|---|---|---|---|
| `TF_NGRAM_AHEAD=0` | 59.3 s | 133,263 | 15.1 GiB |
| this PR | 37.5 s | 14,340 | 2.4 GiB |
| this PR + `madvise(MADV_RANDOM)` on `self.maps` | **35.96 s** | 15,200 | **0.7 GiB** |

Warm: 34.1–34.3 s in all three. Without the PR, the page-cache growth comes from the faults' read-around, and the read-ahead removes most of it. `MADV_RANDOM` on the unlocked maps ([exl3_pack.py#L155](https://github.com/ashhart/TensorFold/blob/dca05e8/src/tensorfold/families/qwen4_exp/cuda/exl3_pack.py#L155)) takes the remaining ~15k faults down to single pages: a further −4.3…−4.9 % and 0.6–0.7 GiB left. The second prompt shows the same pattern (56.0 → 38.1 → 36.3 s). Small side note: the `TF_NGRAM_LOCK` the description mentions is not in this tree.

Data: https://github.com/jschmied/qwen38-flash-next-gb10/blob/2fbd7c5/notes/data/tfreview/ngram.jsonl

Written with AI assistance (Claude Code).
