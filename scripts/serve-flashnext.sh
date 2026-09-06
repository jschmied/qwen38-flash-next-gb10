#!/usr/bin/env bash
# Qwen3.8-Flash-Next launcher -- the live /opt/llm/serve-fnmain.sh from our box, verbatim except FN_MAXLEN,
# which defaults to 32768 here and 8192 there (8192 was a benchmarking choice; one code task emitted 31,115
# characters of thinking before any content). Serving venv: vllm-venv-fnmain2 (nightly dev401 + the #53899
# PLE-offload overlay, tools/main/fnmain-overlay-dev401.diff) plus the four determinism overlays installed by
# tools/main/prod-det-overlays.sh. FN_* variables select model, context, batch, MTP depth, cudagraph mode.
#
# Cautions, both from upstream reports:
#   * num_speculative_tokens=5 hard-fails (QSA ring capacity must divide the attention block size);
#     k=2 and k=3 are the values with evidence behind them.
#   * NEVER combine MTP with --async-scheduling: _prepare_ngram_context reads the CPU token mirror while it
#     still holds speculation's -1 placeholders, giving a wrong n-gram context silently.
# Qwen3.8-Flash-Next, clean bare-metal venv.
#   FN_MTP=k   enable in-checkpoint MTP speculation with k draft tokens (0 = off)
#   FN_SEQS    max concurrent sequences   FN_MAXLEN  context   FN_UTIL  gpu mem frac
#
# Cautions, both from upstream reports:
#   * num_speculative_tokens is constrained, but NOT bounded above. qsa_cache.py
#     get_kv_cache_spec: span = compress_ratio + n; capacity = compress_ratio *
#     cdiv(span, compress_ratio); block_size % capacity == 0 or it hard-fails.
#     With compress_ratio 4 and block_size 848 that makes n = 0..4 and 9..12 legal
#     and n = 5..8 ILLEGAL -- a hole, not a ceiling. (block_size 1600 also allows
#     13..16.) The k=5 failure made 5 look like a wall, so our sweep stopped at
#     k=3; k=4 and the whole 9..12 band are untested. Corrected 2026-08-31.
#   * NEVER combine MTP with --async-scheduling: _prepare_ngram_context reads the
#     CPU token mirror while it still holds speculation's -1 placeholders, giving a
#     wrong n-gram context silently. No benchmark reveals it.
set -euo pipefail
VENV=${FN_VENV:-/opt/llm/runtime/vllm-venv-fnmain}
export HOME=/opt/llm HF_HOME=/opt/llm/hf-cache XDG_CACHE_HOME=${FN_CACHE:-/opt/llm/.cache-fnmain}
export VLLM_CACHE_ROOT=${FN_CACHE_ROOT:-$XDG_CACHE_HOME/vllm} TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=${FN_CACHE_ROOT:-$XDG_CACHE_HOME}/torchinductor
export FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME/flashinfer
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1
# Required for the FlashInfer b12x CuteDSL path (has_flashinfer_b12x_gemm/moe are
# True here); harmless on the cutlass path that AUTO actually picks. NOTE: b12x MoE
# is currently unusable on THIS checkpoint -- the drafter's unquantized MoE vetoes
# it globally (see notes/moe-backend-axis.md). The native `b12x` package, used by
# --linear-backend b12x, is a DIFFERENT package and is not installed in any venv.
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}
# main has no PLE CPU-offload worker: the n-gram tables go to pinned host memory via UVA offload
OFFLOAD=()
export VLLM_PLE_CPU_OFFLOAD=${FN_PLE_OFFLOAD:-1}
# DeepGEMM gates on is_device_capability_family(120), which GB10 (sm_121)
# satisfies -- but its FP8 blockwise kernel faults here with
# "CUDA error: unspecified launch failure" inside deep_gemm.fp8_gemm_nt.
# Same class as the trtllm-gen SM100-only problem. Fall back to the other
# blockwise backend.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
# FP8 GDN projections deterministically hang the engine at c~32 with the default
# cuda kernel (no error, requests stall) -- reported by primitive-ai, bisected.
export VLLM_GDN_DECODE_KERNEL=${VLLM_GDN_DECODE_KERNEL:-triton}
# Determinism fixes (2026-09-06, finding 138 / PR #13 on the #53899 branch), installed on vllm-venv-fnmain2 by
# /opt/llm/runners/prod_det_overlays.sh: deterministic persistent_topk (kernel-det v2.4) and bit-stable MoE finalize
# are env-gated and ON by default here; FN_DET_TOPK=0 / FN_DET_FINALIZE=0 give a stock arm. The PLE offload semaphore
# reset and the FlashInfer autotune cache-key backport are unconditional in the venv.
export VLLM_QSA_DET_TOPK=${FN_DET_TOPK:-1} VLLM_QSA_DET_LIB=${FN_DET_LIB:-/opt/llm/kernel-det/_C_det.so}
export VLLM_MOE_DET_FINALIZE=${FN_DET_FINALIZE:-1}
SPEC=""
# FN_SPEC_MOE gives the DRAFTER its own MoE backend (SpeculativeConfig.moe_backend,
# config/speculative.py:118) -- the documented route for "quantized generator with
# unquantized drafter", so the body can take a backend the drafter cannot.
SPEC_MOE=""
[ -n "${FN_SPEC_MOE:-}" ] && SPEC_MOE=",\"moe_backend\":\"${FN_SPEC_MOE}\""
[ -n "${FN_SPEC_RAW:-}" ] && SPEC="--speculative-config ${FN_SPEC_RAW}"
# Build the JSON from SCALARS, never pass it in whole: systemd Environment= strips
# double quotes, so a JSON blob arrives as {method:ngram,...} and argparse rejects it
# with "cannot be converted to <function loads>". Killed two arms on 2026-08-31.
#   FN_SPEC_METHOD=ngram|ngram_gpu|suffix|mtp   FN_SPEC_N=k
# ngram/suffix are DRAFTLESS and are NOT in EagleModelTypes, so use_eagle() is False
# and the one-block prefix-cache back-off never fires -- unlike mtp/dflash.
if [ -n "${FN_SPEC_METHOD:-}" ]; then
  X=""
  [ -n "${FN_LOOKUP_MAX:-}" ] && X="${X},\"prompt_lookup_max\":${FN_LOOKUP_MAX}"
  [ -n "${FN_SPEC_SHARE:-}" ] && X="${X},\"index_share_for_mtp_iteration\":${FN_SPEC_SHARE}"
  [ -n "${FN_LOOKUP_MIN:-}" ] && X="${X},\"prompt_lookup_min\":${FN_LOOKUP_MIN}"
  [ "${FN_SPEC_NODROP:-0}" = 1 ] && X="${X},\"disable_eagle_block_drop\":true"
  [ "${FN_SPEC_LOCALARGMAX:-0}" = 1 ] && X="${X},\"use_local_argmax_reduction\":true"   # drafter argmax via model.get_top_tokens (reduced draft vocab hook)   # #53388: keep the trailing prefix-cache block
  SPEC="--speculative-config {\"method\":\"${FN_SPEC_METHOD}\",\"num_speculative_tokens\":${FN_SPEC_N:-5}${X}${SPEC_MOE}}"
fi
[ "${FN_MTP:-0}" != "0" ] && SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${FN_MTP}${SPEC_MOE}}"
# Array, not ${VAR:+...}: the JSON contains quotes that word-splitting mangles,
# and systemd Environment= strips quotes, so pass FN_PROF_DIR as a bare path.
PROF=()
if [ -n "${FN_PROF_DIR:-}" ]; then
  PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${FN_PROF_DIR}\"}")
fi
# shellcheck disable=SC2086
exec "$VENV/bin/vllm" serve ${FN_MODEL:-/opt/llm/models/qwen38-flash-next-nvfp4} \
  --served-model-name flashnext --host 127.0.0.1 --port 8092 \
  --max-model-len "${FN_MAXLEN:-32768}" --max-num-seqs "${FN_SEQS:-16}" \
  --max-num-batched-tokens "${FN_BATCH:-4096}" --enable-chunked-prefill \
  --gpu-memory-utilization "${FN_UTIL:-0.90}" \
  --distributed-executor-backend mp --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "${PROF[@]}" "${OFFLOAD[@]}" \
  ${FN_MOE_BACKEND:+--moe-backend $FN_MOE_BACKEND} \
  --compilation-config "{\"cudagraph_mode\":\"${FN_CG_MODE:-PIECEWISE}\",\"cudagraph_capture_sizes\":${FN_CG_SIZES:-[1,2,4,8]}}" \
  ${FN_KVDTYPE:+--kv-cache-dtype $FN_KVDTYPE} \
  ${FN_EXTRA:-} \
  $SPEC
