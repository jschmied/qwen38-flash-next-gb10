"""Env-gated switch of the Flash-Next QSA indexer's block top-k to the deterministic kernel (_C_det, PR #55122 build) on main
(models/qwen4_exp/nvidia/ops/qsa_indexer.py). VLLM_QSA_DET_TOPK=1 activates; VLLM_QSA_DET_LIB=<.so>. `off` removes byte-exactly.
Usage: qsadet_patch2.py [off]   (target overridable with VLLM_QSA_PY)"""
import os, sys
TARGET = os.environ.get("VLLM_QSA_PY") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))), "lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py")
ANCHOR = '''    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
'''
NEW = '''    # ---- QSADET (jschmied 2026-09-06, main port) ----
    import os as _os
    if _os.environ.get("VLLM_QSA_DET_TOPK"):
        if not getattr(_topk, "_qsadet_loaded", False):
            _lib = _os.environ.get("VLLM_QSA_DET_LIB", "/opt/llm/kernel-det/_C_det.so")
            torch.ops.load_library(_lib)
            _topk._qsadet_loaded = True
            print(f"QSADET active: {_lib}", flush=True)
        topk_op = torch.ops._C_det.persistent_topk
    else:
        topk_op = (
            torch.ops._C.cooperative_topk
            if use_cooperative_topk
            else torch.ops._C.persistent_topk
        )
'''
s = open(TARGET).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if NEW not in s: print("  qsadet not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  qsadet REMOVED")
else:
    if NEW in s: print("  qsadet already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  qsadet INSTALLED (inert unless VLLM_QSA_DET_TOPK=1)")
