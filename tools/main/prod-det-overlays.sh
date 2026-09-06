#!/bin/bash
# PROD determinism overlays for vllm-venv-fnmain2 (dev401 + #53899 overlay). Installs (default) or removes (`off`) the four
# fixes from the 2026-09-06 investigation (finding 138 / PR #13 on the #53899 branch):
#   1. qsadet_patch2.py   deterministic persistent_topk (kernel-det v2.4, /opt/llm/kernel-det/_C_det.so), env VLLM_QSA_DET_TOPK=1
#   2. detfin_patch2.py   FlashInfer cutlass MoE use_fused_finalize=False (bit-stable finalize), env VLLM_MOE_DET_FINALIZE=1
#   3. moe_cachekey_patch2.py  FlashInfer MoERunner.get_cache_key_extras (autotune cache key incl. use_fused_finalize) — not env-gated
#   4. plefix_patch.py    PLE offload semaphore reset before each real request — not env-gated
# The envs are defaulted ON in /opt/llm/serve-fnmain.sh (FN_DET_TOPK / FN_DET_FINALIZE=0 to run a stock arm).
# Rules: never while an fx-* unit or a runner is active; one change at a time; remove with this script's `off`.
set -u; S=/opt/llm/runners; V2=/opt/llm/runtime/vllm-venv-fnmain2; P=$V2/lib/python3.12/site-packages/vllm; FI=$V2/lib/python3.12/site-packages/flashinfer
[ "$(systemctl list-units --no-legend 'fx-*' | wc -l)" = 0 ] || { echo "REFUSED: fx-* unit active"; exit 1; }
MODE=${1:-on}; export PYTHONSAFEPATH=1
run(){ echo "-- $1"; shift; "$@" || { echo "   FAILED"; exit 1; }; }
if [ "$MODE" = off ]; then
  run plefix    $V2/bin/python $S/plefix_patch.py off
  run cachekey  env VLLM_FI_CORE_PY=$FI/fused_moe/core.py $V2/bin/python $S/moe_cachekey_patch2.py off
  run detfin    env VLLM_FCM_PY=$P/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py $V2/bin/python $S/detfin_patch2.py off
  run qsadet    env VLLM_QSA_PY=$P/models/qwen4_exp/nvidia/ops/qsa_indexer.py $V2/bin/python $S/qsadet_patch2.py off
else
  [ -f /opt/llm/kernel-det/_C_det.so ] || { echo "missing /opt/llm/kernel-det/_C_det.so"; exit 1; }
  for f in $P/models/qwen4_exp/nvidia/ops/qsa_indexer.py $P/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py $FI/fused_moe/core.py; do [ -f "$f.orig-dev401" ] || cp -a "$f" "$f.orig-dev401"; done
  run qsadet    env VLLM_QSA_PY=$P/models/qwen4_exp/nvidia/ops/qsa_indexer.py $V2/bin/python $S/qsadet_patch2.py
  run detfin    env VLLM_FCM_PY=$P/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py $V2/bin/python $S/detfin_patch2.py
  run cachekey  env VLLM_FI_CORE_PY=$FI/fused_moe/core.py $V2/bin/python $S/moe_cachekey_patch2.py
  run plefix    $V2/bin/python $S/plefix_patch.py
fi
for f in $P/models/qwen4_exp/nvidia/ops/qsa_indexer.py $P/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py $FI/fused_moe/core.py $P/v1/ple_offload/connector.py; do $V2/bin/python -m py_compile "$f" || { echo "COMPILE FAIL $f"; exit 1; }; done
find $P/models/qwen4_exp $P/model_executor/layers/fused_moe $P/v1/ple_offload $FI/fused_moe -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "state: qsadet=$(grep -c VLLM_QSA_DET_TOPK $P/models/qwen4_exp/nvidia/ops/qsa_indexer.py) detfin=$(grep -c VLLM_MOE_DET_FINALIZE $P/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py) cachekey=$(grep -c get_cache_key_extras $FI/fused_moe/core.py) plefix=$(grep -c 'Clear any semaphore' $P/v1/ple_offload/connector.py)"
