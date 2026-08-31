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
# Explicit, ordered list -- NOT a glob. Two reasons:
#  1. hyperconnection.py MUST precede model.py: model.py passes quant_config= to
#     GatedResidual, whose upstream __init__ does not accept it. Applying model.py
#     alone is a TypeError at model construction.
#  2. a glob also picks up experimental/ leftovers and fails on a healthy tree.
PATCHES=(
    v1_ple_offload_connector.py.patch
    v1_worker_gpu_model_runner.py.patch
    models_qwen3_8_flash_next_nvidia_hyperconnection.py.patch
    models_qwen3_8_flash_next_nvidia_model.py.patch
    models_qwen3_8_flash_next_nvidia_mtp.py.patch
    models_qwen3_8_flash_next_nvidia_qsa.py.patch
    models_qwen3_8_flash_next_nvidia_ops_qsa.py.patch
    model_executor_layers_quantization_modelopt.py.patch
)
fail=0
for n in "${PATCHES[@]}"; do
    p="$HERE/$n"
    [ -f "$p" ] || { echo "  MISSING  $n" >&2; fail=1; continue; }
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
# compile() in-process, NOT py_compile: py_compile writes a .pyc beside the source
# regardless of PYTHONDONTWRITEBYTECODE, so as a non-root user it dies on the venv's
# root-owned __pycache__ -- and because it was chained with &&, a genuine syntax error
# was indistinguishable from that permission error. Both were silent. (2026-08-31)
"$VENV/bin/python" - \
    "$SP/vllm/v1/ple_offload/connector.py" \
    "$SP/vllm/v1/worker/gpu/model_runner.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/qsa.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/mtp.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/model.py" \
    "$SP/vllm/models/qwen3_8_flash_next/nvidia/hyperconnection.py" \
    "$SP/vllm/model_executor/layers/quantization/modelopt.py" <<'PYEOF'
import sys
bad = 0
for f in sys.argv[1:]:
    try:
        compile(open(f).read(), f, "exec")
    except SyntaxError as e:
        print(f"  COMPILE FAIL {f}:{e.lineno}: {e.msg}", file=sys.stderr); bad = 1
sys.exit(bad)
PYEOF
if [ $? -eq 0 ]; then echo "  compile check: OK"; else echo "  compile check: FAILED" >&2; fail=1; fi
# the two that fail SILENTLY rather than loudly
grep -q "_input_ready_event" "$SP/vllm/v1/ple_offload/connector.py" \
    && { echo "  WARN: shared _input_ready_event still present (PLE patch not applied)" >&2; fail=1; } \
    || echo "  check: per-request event pool in place"
grep -A4 supported_kv_cache_dtypes "$SP/vllm/models/qwen3_8_flash_next/nvidia/qsa.py" | grep -q fp8_e4m3 \
    && echo "  check: fp8_e4m3 KV advertised" \
    || { echo "  WARN: fp8 KV patch not applied" >&2; fail=1; }
grep -q "quant_config=quant_config" "$SP/vllm/models/qwen3_8_flash_next/nvidia/hyperconnection.py" \
    && echo "  check: hyper-connections receive quant_config" \
    || { echo "  WARN: GatedResidual still hardcodes quant_config=None -- model.py will TypeError" >&2; fail=1; }
exit $fail
