POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/733 (user: "ok do 1")

## What this changes

The native CUDA server accepted three flags it does nothing with, and started as if they had worked:

- `--drafter DIR`: no CUDA family loads a separate drafter. A server started with `--drafter /tmp` came up and drafted with Nemotron's MTP head.
- `--speed-up FILE`: the two-Mac link. No CUDA family reads it.
- `--prompt-cache-over-cap`: on CUDA the prompt cache is clamped to the memory the weights and streams leave, so `--prompt-cache-gib 200 --prompt-cache-over-cap` started with an 87.1 GiB cache and said nothing.

Each is now refused at startup with one line, the way CUDA already refuses `--learn` and Metal refuses `--drafter` for families that don't load one:

```
tensorfold: --drafter loads a separate drafter, and this CUDA family drafts with its own MTP head: serve without --drafter
tensorfold: --speed-up links two Macs: serve without --speed-up on CUDA
tensorfold: --prompt-cache-over-cap lifts a Mac's 70% cap: on CUDA the prompt cache takes at most the memory the weights and streams leave, so serve without it
```

`--drafter` is refused per family. A family that loads a drafter declares `pub const takes_drafter = true`, so the 27B in #608 opts in with one line and Nemotron keeps refusing. The refusals move to `native/cuda_refusal.zig`, `--learn`'s with them, which keeps `cuda.zig` under the 600-line limit.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus this commit; one DGX Spark (GB10, sm_121, driver 580.173.02, CUDA 13); `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`.
- Behaviour: on `0efbd447`, `serve` with `--drafter /tmp`, `--speed-up FILE`, or `--prompt-cache-gib 200 --prompt-cache-over-cap` started and answered. With this commit each exits before loading, with the line above. `--prompt-cache-gib 4` alone still starts.
- Exactness, decode and prompt speed: unchanged (startup checks only; nothing after the model opens changes).
- Tests run: `zig build test` and `zig build test-cuda-host` (25/25 steps), all passing. New tests: "flags with nothing to do on CUDA are refused, not ignored" and "--drafter opens only a family that takes one". Removing the `--speed-up` refusal fails them. `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: Metal (unchanged). The parity survey also suggests Metal Nemotron and Qwen3.5-2B ignore `--speed-up`, but I haven't run that, so this PR leaves Metal alone.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (Nothing after startup changes.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (No bits change.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
