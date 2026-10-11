DRAFT — needs the user's go. ashhart/TensorFold PR from jschmied:prompt-cache-planned-short (2026-10-11). Title: "prompt cache: planned families keep marks on prompts under min_prompt"

This generalises c79c4317b, from #584. There you said the `min_prompt` floor exists for families whose marks end a prompt call early, and that GLM, which keeps only at chunk starts its pass cuts anyway, gained nothing from it, so GLM's store now sets 0. The same holds for every planned family. `min_prompt` (4,096) exists because a mark can cost an extra prompt call. A planned family's marks fall on chunk starts its pass already makes, so for those families the extra call never happens, yet prompts under 4,096 tokens still keep nothing. This PR skips the threshold when `rules.planned` is set. Unplanned families keep it.

- `core/prompt_cache.zig`: `marksRewind` returns no marks below `min_prompt` only when the family is not planned.
- `core/prompt_cache_test.zig`: a planned store keeps its mark on a 10-token prompt, and an unplanned one keeps none. The test fails with the old rule.

Receipts on a GB10 (sm_121, CUDA 13.0, main e7c523eb). CUDA Nemotron 3.5 Lightning MLX-4bit, `tensorfold-native serve --parallel 1`, short agent loop of 12 turns growing from 1.2k to 5.2k tokens, two server starts per arm. The 0 arm sets `min_prompt = 0` through a local switch, which is what this change does for a planned family:

| arm | warm TTFT, turns 2-12 (mean) | turn 1 | replies |
| --- | --- | --- | --- |
| main | 0.39 s (nothing kept until 4,096) | 0.20 s | — |
| min_prompt ignored (planned) | 0.24 s | 0.21 s | equal to main, every turn |

Data: [results3.jsonl](https://github.com/jschmied/qwen38-flash-next-gb10/blob/5803638a205aad504b4415f7dd8bee5a19f06226/notes/data/tf-pcache/results3.jsonl).

On Metal this covers Nemotron, and GLM's explicit `min_prompt = 0` becomes redundant (I left it in). I haven't measured Metal.
