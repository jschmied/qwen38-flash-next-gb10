#!/usr/bin/env bash
# Qwen3.8-Flash-Next, clean bare-metal venv.
#   FN_MTP=k   enable in-checkpoint MTP speculation with k draft tokens (0 = off)
#
# This is the launcher from the box, verbatim except FN_MAXLEN, which defaults to
# 32768 here and 8192 there. 8192 was a benchmarking choice and cannot hold this
# model's own reasoning -- one code task emitted 31,115 characters of thinking
# before any content. Do not copy the 8192.
#   FN_SEQS    max concurrent sequences   FN_MAXLEN  context   FN_UTIL  gpu mem frac
#
# Cautions, both from upstream reports:
#   * num_speculative_tokens=5 hard-fails (QSA ring capacity must divide the
#     attention block size). k=2 and k=3 are the values with evidence behind them.
#   * NEVER combine MTP with --async-scheduling: _prepare_ngram_context reads the
#     CPU token mirror while it still holds speculation's -1 placeholders, giving a
#     wrong n-gram context silently. No benchmark reveals it.
set -euo pipefail
VENV=/opt/llm/runtime/vllm-venv-fnext
export HOME=/opt/llm HF_HOME=/opt/llm/hf-cache XDG_CACHE_HOME=/opt/llm/.cache-fnext
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
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
export VLLM_PLE_CPU_OFFLOAD=1
# DeepGEMM gates on is_device_capability_family(120), which GB10 (sm_121)
# satisfies -- but its FP8 blockwise kernel faults here with
# "CUDA error: unspecified launch failure" inside deep_gemm.fp8_gemm_nt.
# Same class as the trtllm-gen SM100-only problem. Fall back to the other
# blockwise backend.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
# FP8 GDN projections deterministically hang the engine at c~32 with the default
# cuda kernel (no error, requests stall) -- reported by primitive-ai, bisected.
export VLLM_GDN_DECODE_KERNEL=${VLLM_GDN_DECODE_KERNEL:-triton}
SPEC=""
# FN_SPEC_MOE gives the DRAFTER its own MoE backend (SpeculativeConfig.moe_backend,
# config/speculative.py:118) -- the documented route for "quantized generator with
# unquantized drafter", so the body can take a backend the drafter cannot.
SPEC_MOE=""
[ -n "${FN_SPEC_MOE:-}" ] && SPEC_MOE=",\"moe_backend\":\"${FN_SPEC_MOE}\""
[ -n "${FN_SPEC_RAW:-}" ] && SPEC="--speculative-config ${FN_SPEC_RAW}"
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
  "${PROF[@]}" \
  ${FN_MOE_BACKEND:+--moe-backend $FN_MOE_BACKEND} \
  --compilation-config "{\"cudagraph_mode\":\"${FN_CG_MODE:-PIECEWISE}\",\"cudagraph_capture_sizes\":${FN_CG_SIZES:-[1,2,4,8]}}" \
  ${FN_KVDTYPE:+--kv-cache-dtype $FN_KVDTYPE} \
  ${FN_EXTRA:-} \
  $SPEC
