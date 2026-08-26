#!/usr/bin/env bash
# Overlay vLLM PR #53896 (Qwen4Exp model support) onto a venv, Python files only.
# No source build: see notes/why-no-source-build.md.
set -euo pipefail
VENV="${1:?usage: apply-pr53896.sh <venv-path> <staged-head-dir>}"
HEAD="${2:?}"
SP="$VENV/lib/python3.12/site-packages"
n=0
while IFS= read -r -d '' f; do
  rel="${f#$HEAD/}"                       # e.g. vllm/models/qwen4_exp/...
  install -D -m 644 "$f" "$SP/$rel"; n=$((n+1))
done < <(find "$HEAD/vllm" -name '*.py' -print0)
find "$SP/vllm" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
echo "installed $n files into $SP"
