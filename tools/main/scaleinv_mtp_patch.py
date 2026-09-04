"""main-tree port, MTP side: the checkpoint's `lm_head.weight_scale_inv` (FP8_PB_WO, rank-2 [N/128, K/128]) is
mapped onto the MTP head by `_remap_mtp_weight_name`, but `Qwen4ExpMTP.load_weights` lacks the body's
SCALEINV rename/reshape (tools/main/scaleinv_patch.py) -> "no parameter named lm_head.weight_scale_inv" once the
head is quantized (lmhead_mtp_patch.py). Target: vllm/models/qwen4_exp/nvidia/mtp.py. `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_MTP_PY"): return os.environ["VLLM_MTP_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/mtp.py")
TARGET = _target()
ANCHOR = '''        return loader.load_weights(remap_weight_names(), mapper=mapper)
'''
NEW = '''        # ---- SCALEINV-MTP (jschmied 2026-09-04): same rename/reshape as the body loader for
        # FP8_PB_WO `weight_scale_inv` (rank-2 [N/128, K/128] -> `weight_scale` [ob, 1, ib, 1]).
        def _scaleinv(ws):
            for name, w in ws:
                if name.endswith("weight_scale_inv") and w.dim() == 2:
                    yield name[: -len("_inv")], w.reshape(w.shape[0], 1, w.shape[1], 1)
                else:
                    yield name, w
        return loader.load_weights(_scaleinv(remap_weight_names()), mapper=mapper)
        # ---- end SCALEINV-MTP ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  scaleinv-mtp not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  scaleinv-mtp REMOVED")
else:
    if "SCALEINV-MTP" in s: print("  scaleinv-mtp already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  scaleinv-mtp INSTALLED in", TARGET)
