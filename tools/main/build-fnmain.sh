#!/usr/bin/env bash
# Clone vllm-venv-fnext -> vllm-venv-fnmain and install the nightly main wheel (dev352, torch 2.13.0 pin
# satisfied by the clone's 2.13.0+cu130). Clone-don't-build-fresh (see build-027.sh). CPU/disk only.
set -euo pipefail
SRC=/opt/llm/runtime/vllm-venv-fnext; DST=/opt/llm/runtime/vllm-venv-fnmain
WHL=/tmp/claude-1000/-home-jschmied-git-dgx-spark-setup-guide/edab7cc6-31ed-4698-abb3-e11333baf795/scratchpad/wheel/vllm-0.28.1rc1.dev352+gbb363db9a-cp38-abi3-manylinux_2_28_aarch64.whl
[ -d "$DST" ] && { echo "$DST exists — refusing"; exit 1; }
echo "== clone =="; cp -a "$SRC" "$DST"
grep -rl -- "$SRC" "$DST/bin" "$DST/pyvenv.cfg" 2>/dev/null | while read -r f; do [ -f "$f" ] && sed -i "s|$SRC|$DST|g" "$f"; done
find "$DST/lib" -maxdepth 2 -name '*.pth' -exec sed -i "s|$SRC|$DST|g" {} + 2>/dev/null || true
grep -rl -- "$SRC" "$DST/bin" "$DST/pyvenv.cfg" 2>/dev/null && { echo "leftover refs"; exit 1; } || echo "  paths rewritten"
"$DST/bin/python" -c 'import sys; assert sys.prefix.endswith("fnmain"), sys.prefix; print(" prefix", sys.prefix)'
echo "== install main wheel (no deps first) =="; "$DST/bin/pip" install --disable-pip-version-check -q --no-deps "$WHL"
echo "== pip check =="; "$DST/bin/pip" check --disable-pip-version-check || true
echo "== torch still cu130? =="; "$DST/bin/python" -c 'import torch; print(" torch", torch.__version__); assert "+cu130" in torch.__version__'
echo "== import vllm =="; PYTHONSAFEPATH=1 "$DST/bin/python" -c 'import vllm; print(" vllm", vllm.__version__); import vllm._C_stable_libtorch as m; print(" stable ops ok")' 2>&1 | tail -3
echo "== qwen4_exp registered? =="; PYTHONSAFEPATH=1 "$DST/bin/python" -c 'from vllm.model_executor.models.registry import ModelRegistry; print(" ", "Qwen4ExpForConditionalGeneration" in ModelRegistry.get_supported_archs())'
echo "== DONE =="
