DRAFT — the user's go: "do pr asap" (2026-09-27). vllm-project/vllm new PR from jschmied:gdn-recoverssm (base main).
Post only after the rssmup server validation passes (flag path + FULL_AND_PIECEWISE fallback). Title:
[Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv

## Purpose

Speculative decoding on Gated DeltaNet today snapshots the full recurrent state after every draft position, and
reserves one Mamba block per draft position. This PR brings the **RecoverSSM** protocol that Kimi-K3 KDA already uses
to the Qwen GDN layers of Qwen4Exp (Qwen3.8-Flash-Next), plus its PLE dilated short conv:

- **Verify** runs from a read-only checkpoint and stores a small per-token replay record
  `[correction (V), normalized key (K), gate (1)]` in fp32.
- **Commit** (after sampling) replays only the accepted tokens and writes the recurrent state once, plus the
  block-boundary state in align mode. The PLE short conv compacts its accepted window with the same plan.
- `num_speculative_blocks = 0`: no per-draft Mamba blocks.

Enabled like KDA's: `--use-replayssm` together with speculative decoding. Qwen4Exp (CausalLM, ConditionalGeneration,
MTP) declares `SupportsReplaySSM`; without spec decoding the flag is rejected on Qwen4Exp (there is no plain ReplaySSM
path for its GDN/PLE layers). Requires the V2 model runner for align mode, the Triton mamba backend and PP=1 (the same
checks as KDA). The RecoverSSM builders declare `AttentionCGSupport.NEVER`, so the default `FULL_AND_PIECEWISE` falls
back to PIECEWISE with the usual warning; FULL graph support is a follow-up.

Two commits:
1. `[Mamba] Share RecoverSSM commit kernels and helpers across models`: the commit-plan and conv-compaction kernels
   move from `models/kimi_k3` to `model_executor/layers/mamba/ops/recoverssm.py` (Kimi keeps its private names as
   aliases; kernel source unchanged); `mamba/recoverssm_utils.py`; generic request-mapping helpers in
   `v1/attention/backends/recoverssm_metadata.py`; `CacheConfig.use_kda_recoverssm` → `use_recoverssm`. No functional
   change for Kimi-K3.
2. `[Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv`: GDN verify/commit kernels
   (`mamba/gdn/recoverssm_gdn.py`), metadata builders (`gdn_recoverssm.py`, `ple_recoverssm.py`), the layer and model
   integration. The verify reads strided q/k/v views of the conv output and writes straight into the layer output.

Related: #49887 and #56466 (ReplaySSM for GDN) take the ring-buffer route; this PR reuses the merged RecoverSSM
protocol instead (discussed in https://github.com/vllm-project/vllm/pull/56466#issuecomment-5845393579).

## Test Plan

- `tests/kernels/mamba/test_recoverssm_gdn.py`: verify bit-identical to the native fused kernel (checkpoint untouched);
  commit vs the native per-token states (accepted 1–4); align boundaries (8 cases, incl. exact-boundary and crossing);
  mixed prefill/decode mapping and non-contiguous request rows; extreme gates × 3 seeds; strided q/k/v views + `out=`
  (uniform and ragged windows 4/1/3) and the layer's view/fallback decision; BF16 recurrent state (verify exact; commit
  rounds once: within one BF16 ulp of the FP32 reference and no further from it than the native path);
  malformed-input rejection.
- `tests/kernels/mamba/test_recoverssm_ple.py`: spec conv + commit vs the stock token-by-token decode (accepted 1, 2,
  full window; 4 align cases; non-contiguous request rows), bit-identical states and outputs.
- `tests/config/test_recoverssm_config.py`: selection through the real validator (target and the MTP drafter's derived
  config), runtime-check rejections, no spec decoding, unsupported architectures, PIECEWISE fallback declaration.
- Server: Qwen3.8-Flash-Next (NVFP4 experts, FP8 GDN), one DGX Spark GB10 (sm_121), MTP K=3, align mode + prefix
  caching, 2 starts per arm: output hashes, cache-hit replay, decode and agent-loop timing.

## Test Result

Kernel/config tests: <N> passed on GB10 (sm_121). ruff, mypy (tools/pre_commit/mypy.py 3.12), typos: clean.

Server, native GDN spec decode (align) vs RecoverSSM (align), 2 starts per arm
([data](https://github.com/jschmied/qwen38-flash-next-gb10/blob/55687ace78c09d05892bb0663b92838c06459166/notes/speed-of-light.md#L661-L695)):

| | native (align) | RecoverSSM (align) | Δ |
|---|---|---|---|
| c=1 decode, ms per verify cycle | 55.71–56.65 | 54.37–54.38 | −1.3…−2.3 ms |
| c=4 decode, ms per verify cycle | 105.07–105.41 | 99.59–99.91 | −5.2 % |
| agent loop (8 dependent turns), s/turn | 1.31 | 1.10 | −16 % |
| prefix-cache hit rate | 77.4 % | 82.1 % | +4.7 pp |
| KV tokens in the same 4 GiB | 75,678 | 103,953 | +37 % |

- Node trace ([§4u](https://github.com/jschmied/qwen38-flash-next-gb10/blob/55687ace78c09d05892bb0663b92838c06459166/notes/speed-of-light.md#L696-L735)):
  the per-draft state write-back is gone (GDN `out_proj` 101 → 74 µs, no dirty-L2 aftershock); the commit runs at the
  DRAM floor (224 GB/s).
- The zero-copy verify adds −1.2…−1.6 % at c=1 with identical outputs
  ([§5g](https://github.com/jschmied/qwen38-flash-next-gb10/blob/55687ace78c09d05892bb0663b92838c06459166/notes/speed-of-light.md#L1031)).
- Output: cache-hit replay reproduces the first pass 8/8 in every run. Text differs from the native path at the size of a
  reduction-order change (first greedy divergence at a median of 26 tokens, vs 31 for a pure reduction-order change),
  because the recurrent state is now accumulated per commit instead of per token.
- This PR's final shape (flag path, PIECEWISE fallback) on the same box: <rssmup result>.

<details><summary>Essential Elements of an Effective PR Description Checklist</summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>

Developed with AI assistance (Claude Code); all changes reviewed by the author.
