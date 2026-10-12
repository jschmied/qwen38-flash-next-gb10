POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/731 (user: "yes do 1..3")

## What this changes

The follow-up to #654 you agreed to: the GPU sampling rule (1 / T, top_p, the near window of 20, ln(min_p), top_k and the seed) is built once, in `core/lanes/gpu_rule.zig`, next to the host reference it already matches. `gpu_rule.Rule` and `rule(Sampling)` hold the values. `cfg()` and `seedHalves()` give Metal's four floats and two seed halves, and `sample.Rule.from()` gives CUDA's struct.

What now reads it:
- Metal: `Forward.sample` in `families/nemotron/forward.zig`, and `tests/nemotron_sample_check.zig`.
- CUDA: `families/nemotron/cuda_sampler.zig`. `cuda/sample.zig` loses its `rule()`.
- `gpu_rule.sample` and `gpu_full.plan`, the host references.
- `tests/cuda/sample_tests.zig`, which loses its copy of the struct.

The kernels and their arguments are unchanged; the values are the same expressions, so the bits are too. Host tests pin both layouts: Metal's cfg row and seed halves in `gpu_rule`, and `sample.Rule`'s size and field offsets in `cuda/sample.zig`.

#524's `glm/draw_rule.zig` builds the same values again for GLM on Metal; it could fill from `gpu_rule.rule` once both land.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus this commit. CUDA: one DGX Spark (GB10, sm_121, driver 580.173.02, CUDA 13), `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`, built-in kernels. Metal: an M3 Pro, Zig 0.17.0, Xcode 27.0.
- Exactness, CUDA:
  - `tf-cuda-test sample`: 2,880 draws equal the host references.
  - Served Nemotron, `--parallel 1`, base (`0efbd447`) against this branch: 2 prompts x (greedy, and 5 settings x 4 seeds: temperature 0.6 to 1.2, top_p 0.8 to 1.0, top_k 0, 5, 20, 50, min_p 0.05), drafted and `"draft": false`. All 84 replies equal; greedy logprob rows byte-equal (4 of 4); drafted == plain 42 of 42 in both builds. 39 of 40 seeded sampled replies differed from each other, so the draws were live.
- Exactness, Metal: `tf-nemotron-sample-check` on base and branch, "all draws match": 8 cases (flat, peaked, past 1,024 candidates, tied levels, top_k 3000, top_k with top_p, min_p, a draft vocabulary through an id map), 16 of 16 rows each.
- Decode and prompt speed: unchanged (no kernel or launch changes).
- Tests run: `zig build test` 44/44 steps on Linux; on the M3 Pro 154/156 steps, as on base. The one failing step is the same on base: `core.checkpoint_metal` "a split checkpoint keeps later tensors in the file..." logs `checkpoint tensor text.weight stays in its file: read it through a Stage`. New tests: "the rule once: its values, and the Metal cfg row and seed halves it packs", "sample.cu's Rule layout", "a rule in the kernels' layout keeps every field". Swapping two of `cfg()`'s values fails 4 checks. `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: served Nemotron replies on Metal. The Mac here has no Nemotron checkpoint, and it's on a metered connection. If you can run your reply-hash gate on a Mac, that is the missing receipt. RTX cards: no hardware here.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (No bits change.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
