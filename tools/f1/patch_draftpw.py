"""F1: FN_DRAFT_PW=1 keeps PIECEWISE cudagraphs for MTP draft decode steps 2..K (stock forces NONE under PIECEWISE).
Usage: python patch_draftpw.py [on|off]; target via VLLM_X_PY (default: clone venv autoregressive/speculator.py)."""
import os, sys
T = os.environ.get("VLLM_X_PY", "/opt/llm/runtime/vllm-venv-rssm/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py")
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
s = open(T).read()
OLD = """        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
"""
NEW = """        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        elif (__import__("os").environ.get("FN_DRAFT_PW", "") == "1"  # FNDRAFTPW
              and cudagraph_mode.decode_mode() == CUDAGraphMode.PIECEWISE):  # FNDRAFTPW
            cudagraph_mode = CUDAGraphMode.PIECEWISE  # FNDRAFTPW
            logger.warning("FNDRAFTPW: PIECEWISE cudagraphs for MTP draft decode steps")  # FNDRAFTPW
        else:
            cudagraph_mode = CUDAGraphMode.NONE
"""
if MODE == "on":
    if "FNDRAFTPW" in s: sys.exit("already on")
    if s.count(OLD) != 1: sys.exit(f"anchor count {s.count(OLD)}")
    s = s.replace(OLD, NEW)
else:
    if s.count(NEW) != 1: sys.exit("remove anchor missing")
    s = s.replace(NEW, OLD)
open(T, "w").write(s); print("FNDRAFTPW", MODE, T)
