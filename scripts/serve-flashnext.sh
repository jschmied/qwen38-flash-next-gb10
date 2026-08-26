#!/usr/bin/env bash
# Qwen3.8-Flash-Next prototype. NO speculative decoding on purpose: the only
# C-level change in vLLM PR #53896 is a signature extension to
# fused_gdn_decode_post_conv_mtp (adds output_gate_activation, which Qwen4
# needs as 'sigmoid'). That op is gated on attn_metadata.num_spec_decodes > 0,
# so with speculation off the stock prebuilt kernel is never called and no
# source build is required.
set -euo pipefail
cd /opt/llm
VENV=/opt/llm/runtime/vllm-venv-flashnext
export HOME=/opt/llm HF_HOME=/opt/llm/hf-cache XDG_CACHE_HOME=/opt/llm/.cache-flashnext
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
export FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME/flashinfer
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTE_DSL_ARCH=sm_121a VLLM_MARLIN_USE_ATOMIC_ADD=1
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1        # bounded JIT fan-out (OOM guard)
mapfile -t KEYS < <(grep -vE '^[[:space:]]*([#;]|$)' /etc/llama-server/api_keys.txt)
exec "$VENV/bin/vllm" serve /opt/llm/models/qwen38-flash-next-nvfp4 \
  --served-model-name flashnext --host 127.0.0.1 --port 8092 --api-key "${KEYS[@]}" \
  --trust-remote-code \
  --max-model-len "${FN_MAXLEN:-8192}" --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2]}' \
  --gpu-memory-utilization "${FN_UTIL:-0.90}" \
  --distributed-executor-backend mp \
  ${FN_EXTRA:-}
