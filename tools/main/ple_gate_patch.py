"""main-tree port of the preview PLE gate widening: ModelOpt (mixed/fp4) checkpoints ship the PLE as
F8_E4M3 shards + one global `ngram_embedding.weight_scale`, exactly what Qwen4ExpPLEFp8EmbeddingMethod
implements, but the isinstance(Fp8Config) gate rejects them. Target: the vLLM of the running interpreter
(override VLLM_PLE_PY). `off` removes byte-exactly."""
import os, sys
def _target():
    if os.environ.get("VLLM_PLE_PY"): return os.environ["VLLM_PLE_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/ple_layer.py")
TARGET = _target()
ANCHOR = "    if not isinstance(quant_config, Fp8Config):\n        return None\n"
NEW = '''    if not isinstance(quant_config, Fp8Config):
        # ---- PLEGATE (jschmied 2026-09-03, port of the 2026-08-27 preview patch) ----
        _name = ""
        try:
            _name = quant_config.get_name()
        except Exception:
            pass
        if _name in ("modelopt", "modelopt_fp4", "modelopt_mixed", "modelopt_mxfp8"):
            return Qwen4ExpPLEFp8EmbeddingMethod()
        # ---- end PLEGATE ----
        return None
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  plegate not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  plegate REMOVED")
else:
    if "PLEGATE" in s: print("  plegate already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  plegate INSTALLED in", TARGET)
