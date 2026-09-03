"""main-tree port of the FP8_PB_WO `weight_scale_inv` convention (ModelOpt 0.46 export: rank-2
[N/128, K/128] named `weight_scale_inv`; vLLM's KFp8PbWo creates `weight_scale` as [ob,1,ib,1]).
Renames and reshapes at load time in Qwen4ExpForConditionalGeneration.load_weights. `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_MODEL_PY"): return os.environ["VLLM_MODEL_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/model.py")
TARGET = _target()
ANCHOR = '''        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)
'''
NEW = '''        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        # ---- SCALEINV (jschmied 2026-09-03): ModelOpt FP8_PB_WO exports name the block
        # scale `weight_scale_inv` (rank-2 [N/128, K/128]); vLLM expects `weight_scale`
        # shaped [ob, 1, ib, 1]. Same quantity, "_inv" is a naming legacy (verified).
        def _scaleinv(ws):
            for name, w in ws:
                if name.endswith("weight_scale_inv") and w.dim() == 2:
                    yield name[: -len("_inv")], w.reshape(w.shape[0], 1, w.shape[1], 1)
                else:
                    yield name, w
        weights = _scaleinv(weights)
        # ---- end SCALEINV ----
        return loader.load_weights(weights, mapper=mapper)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  scaleinv not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  scaleinv REMOVED")
else:
    if "SCALEINV" in s: print("  scaleinv already installed"); raise SystemExit
    assert s.count(ANCHOR) >= 1, "anchor"
    # patch the LAST occurrence: Qwen4ExpForConditionalGeneration.load_weights (the served class)
    i = s.rfind(ANCHOR)
    open(TARGET, "w").write(s[:i] + NEW + s[i + len(ANCHOR):]); print("  scaleinv INSTALLED (last load_weights) in", TARGET)
