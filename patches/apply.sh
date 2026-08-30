#!/usr/bin/env bash
# Re-apply every local vLLM modification this box depends on.
# ANY pip install/upgrade of vLLM in this venv silently reverts all of them; the
# symptoms are non-obvious (startup hang at warmup_kernels, HTTP 400 on tools,
# missing scale parameters), so run this after every reinstall.
set -Eeuo pipefail
VENV="${VENV:-/opt/llm/runtime/vllm-venv-fnext}"
SP="$VENV/lib/python3.12/site-packages"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$SP/vllm" ] || { echo "error: no vllm in $SP" >&2; exit 2; }
ver="$("$VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || true)"
EXPECT="0.1.dev20073+g8e685d198"
if [ "$ver" != "$EXPECT" ] && [ "${ALLOW_OTHER_VERSION:-0}" != "1" ]; then
    echo "error: these patches were cut against $EXPECT, found '$ver'." >&2
    echo "       Re-cut them rather than forcing; upstream renamed the package" >&2
    echo "       qwen3_8_flash_next -> qwen4_exp and refactored ple_layer.py." >&2
    exit 2
fi
fail=0
for p in "$HERE"/*.patch; do
    n=$(basename "$p")
    if patch -d "$SP" -p1 --forward --dry-run -i "$p" >/dev/null 2>&1; then
        patch -d "$SP" -p1 --forward -i "$p" >/dev/null && echo "  applied  $n"
    elif patch -d "$SP" -p1 --reverse --dry-run -i "$p" >/dev/null 2>&1; then
        echo "  already  $n"
    else
        echo "  FAILED   $n" >&2; fail=1
    fi
done
# Compile-check rather than import: the model modules are circular by design and
# only resolve when vLLM loads them through the package, so a bare import of
# nvidia/qsa.py raises ImportError on a perfectly good tree.
PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -m py_compile \
    "$SP/vllm/v1/ple_offload/connector.py" \
    "$SP/vllm/v1/worker/gpu/model_runner.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/qsa.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/mtp.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/model.py" \
    "$SP/vllm/model_executor/layers/quantization/modelopt.py" && echo "  compile check: OK"
# the two that fail SILENTLY rather than loudly
grep -q "_input_ready_event" "$SP/vllm/v1/ple_offload/connector.py" \
    && { echo "  WARN: shared _input_ready_event still present (PLE patch not applied)" >&2; fail=1; } \
    || echo "  check: per-request event pool in place"
grep -A4 supported_kv_cache_dtypes "$SP/vllm/models/qwen3_8_flash_next/nvidia/qsa.py" | grep -q fp8_e4m3 \
    && echo "  check: fp8_e4m3 KV advertised" \
    || { echo "  WARN: fp8 KV patch not applied" >&2; fail=1; }
exit $fail
