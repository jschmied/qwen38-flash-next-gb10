DRAFT — needs the user's go. ashhart/TensorFold new issue (2026-10-11). Title: "CUDA Nemotron: the prompt path is chunk-invariant, so the cache's 2,048 grid could go"

Since e2dcb781, CUDA Nemotron's prompt cache keeps and resumes only on the 2,048-row grid (`.planned = true, .grid = prompt_rows` in [native/cuda.zig](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/src/native/cuda.zig#L450), [cuda_prompt_grid.zig](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/src/families/nemotron/cuda_prompt_grid.zig)). On a GB10 (sm_121, CUDA 13.0, main e7c523eb, Nemotron 3.5 Lightning MLX-4bit), the prompt path's bits don't depend on where the chunks are cut, so the grid costs reuse without buying exactness.

**Chunk invariance.** `tensorfold prefill` of a 5,000-token prompt with `--dump`, the first cut moved by a local switch:

| chunks | all 53 residual dumps and logits vs the grid run (2048/2048/904) |
| --- | --- |
| 1500 / 2048 / 1452 | byte-equal |
| 7 / 2048 / 2048 / 897 | byte-equal |
| 1000 / 2048 / 1952 | byte-equal |
| 894 / 2048 / 2048 / 10 | byte-equal |

**Agent loop.** `tensorfold-native serve --parallel 1`, a 7.9k-token start, then 12 turns that each add the reply (96 tokens, temperature 0) and about 1.1k tokens of tool output. Two server starts per arm:

| arm | where turn 12 resumes | warm TTFT, turns 2-12 | replies vs cache off |
| --- | --- | --- | --- |
| main (grid) | 18,432 of 20,799 | 0.23-0.51 s, mean 0.395 | equal |
| grid removed | 19,748 of 20,799 | 0.24-0.35 s, mean 0.306 | equal |
| cache off | — | mean 2.23 s | — |

Removing the grid took four local changes:
- the Rules without `planned`/`grid`;
- `resumable` accepting any row;
- `serialChunks` also ending a chunk at a keep mark;
- the head's last row computed for a last chunk that starts off the grid (`(len - 1) % prefill_rows` assumes the grid).

The first turn costs more (1.14 to 1.35 s), because each mark ends a chunk, and the cache holds more states (15 / 4.4 GiB vs 9 / 2.6 GiB).

Not tested: the NVFP4 checkpoint, staggered segments, `--slide` lessons, Metal.

Data and the experiment's diff: [results](https://github.com/jschmied/qwen38-flash-next-gb10/blob/82d8d75c1325a2860e3603dbdabc3997570054af/notes/data/tf-pcache), [diff](https://github.com/jschmied/qwen38-flash-next-gb10/blob/82d8d75c1325a2860e3603dbdabc3997570054af/notes/data/tf-pcache/nogrid-experiment.diff).

If you'd like it, I can open the four changes as a PR with a test: a resumed pass equals a fresh one at off-grid marks.
