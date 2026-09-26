"""A2 hook: FN_MTP_DENSE_NVFP4=1 swaps the MTP drafter's BF16 dense linears to fn_nvfp4_dense after loading.
Usage: python patch_mtpdense4.py [on|off]; target via VLLM_X_PY (default: the clone venv's mtp.py)."""
import os, sys
T = os.environ.get("VLLM_X_PY", "/opt/llm/runtime/vllm-venv-rssm/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/mtp.py")
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
s = open(T).read()
OLD = "        return loader.load_weights(_scaleinv(remap_weight_names()), mapper=mapper)\n"
NEW = """        _fn_loaded = loader.load_weights(_scaleinv(remap_weight_names()), mapper=mapper)  # FNMTPDENSE4
        if __import__("os").environ.get("FN_MTP_DENSE_NVFP4", "") == "1":  # FNMTPDENSE4
            from vllm.models.qwen4_exp.nvidia.fn_nvfp4_dense import swap_methods  # FNMTPDENSE4
            swap_methods(self)  # FNMTPDENSE4
        return _fn_loaded  # FNMTPDENSE4
"""
if MODE == "on":
    if "FNMTPDENSE4" in s: sys.exit("already on")
    if s.count(OLD) != 1: sys.exit(f"anchor count {s.count(OLD)}")
    s = s.replace(OLD, NEW)
else:
    if s.count(NEW) != 1: sys.exit("remove anchor missing")
    s = s.replace(NEW, OLD)
    if "FNMTPDENSE4" in s: sys.exit("leftover marker")
open(T, "w").write(s); print("FNMTPDENSE4", MODE, T)
