DRAFT — needs the user's go. ashhart/TensorFold PR, base `pr-141-0.6.1`, head `jschmied:exl3-tile-list-061` (2026-10-01).
Title: CUDA EXL3 experts: one program a member tile in use, not one a tile of the window (prompts −10 % on Flash Next)

The grouped EXL3 expert kernels launch ⌈maxm / 16⌉ member tiles for every expert, where maxm is the window's rows. On
Flash Next's 1,024-row prompt windows, a routed expert has about 20 rows (2 tiles) but gets 64 programs, and almost all
of them return after reading `members`. A window's shared expert holds every row, so maxm can't shrink on the real
path. Nsight Compute on GB10: 1.3 M gate/up programs a window, 12–14 % memory throughput, 17–19 % SM throughput.

The grouping now also lists each expert's non-empty 16-row tiles (`(place << 16) | tile`, and their count). The grouped
kernel takes its expert and member tile from that list. The grid becomes (⌈R · slots / 16⌉ + experts, N blocks,
matrices · splits), a bound the host knows without a sync. Each program's arithmetic is unchanged, so outputs are
bit-identical. The second scan adds 128 bytes of static shared memory; the large-launch check counts it.

## Measured

One GB10, Flash Next EXL3 3.05 bpw (512 routed experts + the shared one, 11 slots), two or three alternating runs per
cell, on 0.6.0 plus this change. On `pr-141-0.6.1` the routed() cells repeat below.

| | before | after |
|---|---|---|
| `routed()`, 1,024 rows (ms) | 26.42 / 26.39 / 26.41 | 18.69 / 18.67 / 18.70 (−29 %) |
| `routed()`, 64 / 8 / 1 rows | 3.87 / 0.75 / 0.077 | 3.57 / 0.75 / 0.079 |
| 8k prompt, prefill (s) | 11.42 / 11.47 | 10.35 / 10.32 (−9.4…−10.0 %) |
| 32k prompt, prefill (s) | 46.18 / 46.38 | 41.79 / 41.66 (−9.5…−10.2 %) |

Bit-identical: `routed()` output hashes at 1, 8, 64 and 1,024 rows, and the first 16 tokens after both prompts.

## Tests

- `tests/cuda/test_exl3_experts.py`: new test that the grouping lists exactly every expert's ⌈members / 16⌉ tiles in
  place order, with a shared expert holding every row.
- `tests/cuda/test_group_kernel_smem.py`: the calls pass the tile list and check it too, and the opt-in launch leaves
  room for the kernel's 256 static bytes instead of 128.
- On GB10: test_exl3_experts, test_qwen4_exp_exl3 and test_group_kernel_smem: <RESULT>.

Not measured: GLM and Qwen3.5 EXL3 checkpoints (`routed()` is shared, the change is shape-independent), and two ranks.

Written with AI assistance (Claude Code); the author reviewed every change.
