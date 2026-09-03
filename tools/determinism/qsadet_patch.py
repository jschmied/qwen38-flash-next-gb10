"""Env-gated switch of the QSA block selection to the deterministic kernel (patches/kernel-det),
built standalone as `_C_det` (see patches/kernel-det/build_det.py).
  VLLM_QSA_DET_TOPK=1        use torch.ops._C_det.persistent_topk instead of _C.persistent_topk
  VLLM_QSA_DET_LIB=<path>    the .so (default /opt/llm/kernel-det/_C_det.so)
Inert without VLLM_QSA_DET_TOPK. Composes with qsafix_patch.py (VLLM_QSA_EXACT_TOPK wins when set).
`off` removes byte-exactly."""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
ANCHOR = '''        topk_op = (
            torch.ops._C.cooperative_topk
            if use_cooperative_topk
            else torch.ops._C.persistent_topk
        )
'''
NEW = '''        # ---- QSADET (jschmied 2026-09-03) ----
        import os as _os
        if _os.environ.get("VLLM_QSA_DET_TOPK"):
            if not getattr(qsa_select_paged_tokens, "_qsadet_loaded", False):
                _lib = _os.environ.get("VLLM_QSA_DET_LIB", "/opt/llm/kernel-det/_C_det.so")
                torch.ops.load_library(_lib)
                qsa_select_paged_tokens._qsadet_loaded = True
                print(f"QSADET active: {_lib}", flush=True)
            topk_op = torch.ops._C_det.persistent_topk
        else:
            topk_op = (
                torch.ops._C.cooperative_topk
                if use_cooperative_topk
                else torch.ops._C.persistent_topk
            )
        # ---- end QSADET ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  qsadet not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  qsadet REMOVED")
else:
    if "QSADET" in s: print("  qsadet already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  qsadet INSTALLED (inert unless VLLM_QSA_DET_TOPK=1)")
