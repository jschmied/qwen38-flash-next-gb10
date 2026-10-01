POSTED as #211 (2026-10-02). ashhart/TensorFold PR, base main, head jschmied:flashnext-fp4-experts (99ccde3).
Title: Flash Next CUDA: routed NVFP4 experts in the checkpoint's own math (FP4 x FP4), prompts −9…−10 %

Follow-up to #173. In 0.6.1 `--precision checkpoint` reaches the 27B only: `precision.mode()` and
`cuda/nvfp4/checkpoint.py` are used by `families/qwen3_5`, and Flash Next's routed experts still run
`nvfp4_expert_kernel` on bf16 rows whatever the flag. This puts them under the same contract.

- Rows enter gate|up once a layer in NVFP4 (`quant4`) under the experts' static input scale. The checkpoint stores
  one per expert; ours holds a single value for all 512 in every layer but layer 0 (two values), so the layer takes the
  largest, as FlashInfer's CUTLASS MoE does.
- `cuda/nvfp4/experts_ck.cu`: a warp per (plan item, 32 columns) on the block-scaled FP4 mma (`mma4.cuh`), against
  the lane matmul's words and block scales (`pack4`) stacked per projection and re-laid lane-major, four K steps in
  flight. gate|up's epilogue is SiLU(gate) * up in fp32, quantized to down's NVFP4 input under each expert's own down
  input scale (`nvfp4q.cuh`, as `gemm_gu_ck` does); down writes each pair's row in fp32 or bf16.
- One fp32 chain over K a pair, so a pair's bits never depend on the others: drafted == serial and resumed == fresh
  hold. Same weight bytes as the W4A16 blocks.
- One rank on SM 12.x. `--precision full`, two ranks and other GPUs keep `nvfp4_expert_kernel`; so do the MTP
  layer's draft-only experts. The startup line names the experts' math. A checkpoint without the experts' input scales
  is refused under `checkpoint` (as the 27B's loader does).

## Measured

One DGX Spark (GB10), our ModelOpt export (NVFP4 routed experts, block-FP8 dense layers), `--precision full` vs
`checkpoint` on this branch, one process an arm
([data](https://github.com/jschmied/qwen38-flash-next-gb10/blob/9e7c4c4d348b7e533a72f94c7d03c8237e79eb4a/notes/data/tffp4x)):

| | full | checkpoint | |
|---|---|---|---|
| 8k prompt, prefill (s), 3 alternating rounds | 6.11 / 6.13 / 6.18 | 5.57 / 5.60 / 5.56 | −8.7…−9.9 % |
| 32k prompt, prefill (s) | 24.51 / 24.62 / 24.78 | 22.32 / 22.41 / 22.33 | −8.9…−9.9 % |
| 8k prompt, routed experts' kernels | 2,154 ms | 1,597 ms | −26 % |
| decode, 8 prompts x 256 greedy tokens, drafts on | 52.45 tok/s | 52.1 tok/s | −0.6 % |

The first 16 tokens after both prompts are the same in both modes. Decode rounds are 3 % faster (54.6 vs 56.3 ms) and
carry 3.8 % fewer tokens: the replies differ, and so do their drafts.

Quality, teacher-forced, 8 sequences of 4,096 positions (wikitext-2 test x 4, CPython source x 4), the decode path,
against `--precision full` (Flash Next has no fp32 forward):

| | KL(full ‖ checkpoint) | top-1 agreement | perplexity |
|---|---|---|---|
| wikitext | 0.051 | 93.0 % | +1.29 % |
| code | 0.026 | 98.3 % | +1.26 % |
| all | 0.038 | 95.6 % | +1.27 % |

KL takes the reference's top 32 tokens and the rest as one bucket.

## Tests

- `tests/cuda/test_nvfp4_experts_ck.py`: gate|up's NVFP4 bytes against a torch quantizer of the fp64 SwiGLU (fewer
  than 1 % of bytes differ, where an fp32 sum sits on an e2m1 edge); down equals the fp64 product of its NVFP4 rows to
  1e-5, bf16 = fp32 rounded; a pair's bits alone and in decode or prompt plans (16- and 64-pair items); the scales
  `make_ck` takes.
- `tests/cuda/test_flashnext_nvfp4_loader.py`: the W4A16 face tests pin `--precision full`; a new one loads the tiny
  checkpoint under `checkpoint`.
- On GB10: `tests/cuda` 1,208 passed, 98 skipped. Drafted == serial on the real checkpoint under `checkpoint`: 4 of
  4 prompts, 160 tokens each.

Not run: two ranks, other SM 12.x cards, RadixArk's and local-inference-lab's exports (both store per-expert input
scales; I haven't checked whether gate/up's are uniform there).

Written with AI assistance (Claude Code); the author reviewed every change.
