"""Install/remove FNCAP in the main venv: copy vllm/fn_capture.py and hook it after model load in the V2 runner.
Inert unless FN_CAPTURE_DIR is set. `off` removes both byte-exactly. Optional argv[2]: dir with a model_runner.py copy."""
import os, shutil, sys
PKG = "/opt/llm/runtime/vllm-venv-main1ea7/lib/python3.12/site-packages/vllm"
MR = os.path.join(sys.argv[2], "model_runner.py") if len(sys.argv) > 2 else f"{PKG}/v1/worker/gpu/model_runner.py"
MOD_DST = os.path.join(sys.argv[2] if len(sys.argv) > 2 else PKG, "fn_capture.py")
MOD_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_capture.py")
A = """            time_after_load - time_before_load,
        )

"""
N = """            time_after_load - time_before_load,
        )
        # ---- FNCAP (jschmied 2026-09-24): env-gated tensor capture for kernel replay ----
        if __import__("os").environ.get("FN_CAPTURE_DIR"):
            from vllm import fn_capture as _fn_capture
            _fn_capture.install(self.model, getattr(getattr(self, "speculator", None), "model", None))
        # ---- end FNCAP ----

"""
s = open(MR).read()
if sys.argv[1:2] == ["off"]:
    if "FNCAP" in s:
        s = s.replace(N, A); assert "FNCAP" not in s; open(MR, "w").write(s); print("  fncap REMOVED")
    if os.path.exists(MOD_DST): os.remove(MOD_DST)
else:
    if "FNCAP" not in s:
        assert s.count(A) == 1, "anchor"; open(MR, "w").write(s.replace(A, N)); print("  fncap INSTALLED")
    shutil.copyfile(MOD_SRC, MOD_DST)
