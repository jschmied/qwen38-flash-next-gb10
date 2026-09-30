POSTED 2026-09-30 as https://github.com/ashhart/TensorFold/pull/126 — user's go 2026-09-30 ("do all now", after "can we open PR to tensorfold with our changes?"). ashhart/TensorFold PR from jschmied/TensorFold:fp8-block into main.

TITLE: CUDA: read block-scaled FP8 linears (ModelOpt FP8_PB_WO) in Flash Next checkpoints

Flash Next checkpoints whose dense linears ship as block-scaled FP8 (ModelOpt `FP8_PB_WO`, the DeepSeek layout:
e4m3 bytes and an fp32 `weight_scale_inv` per 128×128 block) now load on the CUDA route and decode from their stored
bytes. Before, the family's check refused `FP8_PB_WO`, and the loader would have read those tensors as per-tensor FP8
(or, for `lm_head`, cast the e4m3 bytes to bf16 without their scale).

## The format and the route

- **Decode: a fourth lane-matmul mode, `FP8G`.** The e4m3 bytes sit in the FP8 GEMM's fragment order (as `Fp8Linear`),
  and each (64 inputs, column) gets its block's scale as one fp32. A 64-input stage is multiplied in bf16 MMAs, since
  e4m3 fits bf16 exactly, and the stage's products are added into the accumulator times their scale in stage order.
  The stored weight is used exactly, and a row's bits do not depend on the row count, the same contract as the NVFP4
  and MXFP8 modes. The scale grid also expresses per-channel and per-tensor FP8.
- **Prompts** run the FP8 prompt matmul (`qmm_prefill8w`) over the same bytes, with the block scales as bf16 group
  scales. Rounding them to nearest bf16 moves a weight by at most 2⁻⁹ relative, less than the prompt path's own e4m3
  rounding of the activations. Changing the shared prompt kernel to fp32 scales looked like more than this PR should
  carry. If you'd rather have it exact, I can do that.
- **`Fp8BlockLinear`** (`cuda/nvfp4/linear.py`) has the same face as `Mx8Linear`: `__call__` for decode, `prefill`
  for prompts.
- **Mixed stacks.** A projection stack that mixes block FP8 with bf16 runs each part on its own kernel into its
  columns (`Concat`). Examples: `in_proj_b` and `in_proj_a` stay bf16 beside the FP8 `in_proj_qkv` and `in_proj_z`,
  and the indexer's projection stays bf16 beside the FP8 q/k/v.
- **A block-FP8 `lm_head`** is dequantized to bf16 at load, in row chunks (code × block scale). That covers both the
  head and the draft head's rows.
- **Loader and checks.** `format.scheme` names the layout `fp8block` (`weight_scale_inv`, 2-D fp32), `format.dequant`
  gains its numpy dequantizer, and the family's config check accepts `FP8_PB_WO`. `--tp 2` on block FP8 stops by
  name, as the NVFP4 route does.

## Tests

- `tests/cuda/test_nvfp4_linear.py`:
  - decode against an fp64 reference from `format.dequant`, including N not a multiple of 128;
  - rows alone and in 2/3/5/12/16-row windows;
  - prompt chunks bit-equal;
  - `Concat` equal to its parts.
- `tests/test_nvfp4_format.py`: the numpy dequantizer against torch's float8, and the scheme from tensor storage.
- `tests/cuda/nvfp4_tiny.py` gains an `fp8block` layout: block FP8 for the DeltaNet qkv/z/out and attention q/k/v/o
  projections and `lm_head`, bf16 for the rest, so stacks mix.
- `tests/cuda/test_flashnext_nvfp4_loader.py`:
  - the loader puts each linear on its kernel, and the head equals the block dequant;
  - the engine decodes;
  - drafts over a draft vocabulary keep the serial tokens on the new layout.

On one DGX Spark (GB10, sm_121), torch 2.13 + cu130, Triton 3.7.1:
- the new and existing NVFP4 tests: `test_nvfp4_format.py`, `cuda/test_nvfp4_linear.py`,
  `cuda/test_flashnext_nvfp4_loader.py`, 39 passed;
- the other CUDA tests through the lane matmul (`test_flashnext_nvfp4_kernels.py`, `test_flashnext_nvfp4.py`,
  `test_nvfp4_experts.py`, `test_qwen27_nvfp4.py`): 36 passed, 3 skipped.

## Served

A local ModelOpt export of Qwen3.8-Flash-Next on one DGX Spark (GB10, 128 GB), `tensorfold serve … --context 32768`.
The export has NVFP4 experts; block-FP8 DeltaNet, attention and `lm_head`; an FP8 n-gram table; and NVFP4 MTP experts.

- **Exactness.** Replies were compared by the server's `token_sha`:
  - 6 pairs of ~2k/8k/16k-token prompts, greedy and sampled (seed): drafted replies equal `"draft": false` ones, and
    a resumed repeat equals the fresh run;
  - `tools/bench_concurrent.py --levels 1,2,4 --alone --serial` (code and chat, greedy and sampled): every concurrent
    reply equals its request alone (3/3, 6/6, 12/12), and every solo run equals serial.
- **Against the same weights on the bf16 path.** The block-FP8 linears were dequantized to bf16 at load (a local
  switch, not in this PR): same checkpoint, drafts on, one start each. Decode tok/s:

  | | block FP8 (this PR) | bf16 |
  |---|---|---|
  | `bench_concurrent` code, c=1, sampled / greedy | 60.1 / 59.2 | 52.0 / 49.1 |
  | `bench_concurrent` chat, c=1, sampled / greedy | 41.6 / 36.6 | 32.8 / 34.6 |
  | `bench_openai` 400 tokens, fibonacci / chat, greedy | 59.4 / 37.6 | 49.3 / 32.3 |
  | `bench_openai` 64 tokens, fibonacci / chat, greedy | 48.7 / 40.2 | 45.2 / 36.0 |

  The bf16 copy rounds each weight, so the two columns decode different replies; both pass the same exactness
  checks.
- **Decode matmul alone** (real projections, 50 warm-up and 200 timed calls, CUDA events). `in_proj_qkv`
  [10240, 2560] takes 94 µs on FP8G against 247 µs on the bf16 matmul, and `q_proj` [12288, 2560] 128 against
  292 µs, flat from 1 to 8 rows. 94 µs is about the chip's DRAM floor for those 26 MB.

## Not in this PR

- **`nvidia/Qwen3.8-Flash-Next-NVFP4` is still refused.** Its MTP experts are `FP8_BLOCK_SCALES` (e4m3 with bf16
  `weight_scale_inv` per 128×128 block), and its n-gram table is labelled `FP8`. The block format here is most of
  what that needs. The MTP experts only draft, so dequantizing them would not change a reply's bits. I have not
  loaded that checkpoint, so it is not claimed here.
- **Per-channel FP8 from compressed-tensors** (e.g. `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8`) fits the same
  scale grid. This route does not read compressed-tensors Flash Next checkpoints yet.
- **`Concat` costs about 45 µs a layer** at decode widths: one bf16 matmul for the two small rows plus two
  column copies. An output stride on the lane matmul would remove the copies.

Written with AI assistance (Claude Code); the author reviewed every change.
