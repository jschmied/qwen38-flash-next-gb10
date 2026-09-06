"""Env-gated deterministic finalize for FlashInfer's CUTLASS MoE on main: VLLM_MOE_DET_FINALIZE=1 passes
use_fused_finalize=False (non-fused, deterministic finalize; no atomics). Usage: detfin_patch2.py [off]; target via VLLM_FCM_PY."""
import os, sys
TARGET = os.environ.get("VLLM_FCM_PY") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))), "lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py")
ANCHOR = '''            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            use_w4_group_scaling=use_w4_group_scaling,
        )
'''
NEW = '''            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            use_w4_group_scaling=use_w4_group_scaling,
            # ---- DETFIN (jschmied 2026-09-06, main port): non-fused deterministic finalize on demand
            use_fused_finalize=not bool(__import__("os").environ.get("VLLM_MOE_DET_FINALIZE")),
        )
'''
s = open(TARGET).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if NEW not in s: print("  detfin not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  detfin REMOVED")
else:
    if NEW in s: print("  detfin already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  detfin INSTALLED (inert unless VLLM_MOE_DET_FINALIZE=1)")
