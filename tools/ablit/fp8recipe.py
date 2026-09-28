"""Reproduce our checkpoint's FP8 o_proj from RadixArk's BF16 (blockwise 128x128 E4M3, scale = amax/448,
stored as weight_scale_inv = scale). Bit-exact check against our mtpfp4 tensors. Prints per-layer results."""
import json, sys, torch
from safetensors import safe_open
D = "/opt/llm/models/qwen38-flash-next-mtpfp4"; S = sys.argv[1]
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def ours(n):
    with safe_open(f"{D}/{wm[n]}", "pt") as f: return f.get_tensor(n)
def quant(w, variant):
    R, C = w.shape; b = w.float().reshape(R // 128, 128, C // 128, 128).permute(0, 2, 1, 3)
    amax = b.abs().amax(dim=(2, 3))
    scale = amax / 448.0 if variant == "div" else amax * (1.0 / 448.0)
    q = (b / scale[:, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.permute(0, 2, 1, 3).reshape(R, C), scale
res = {}
with safe_open(f"{S}/radix_oproj.safetensors", "pt") as rx:
    for n in rx.keys():
        w = rx.get_tensor(n); o = ours(n)
        if o.dtype == torch.bfloat16:
            res[n] = {"ours_dtype": "bf16", "ours_equals_radix": bool(torch.equal(o, w))}; continue
        si = ours(n + "_scale_inv")
        r = {}
        for v in ("div", "mul"):
            q, s = quant(w, v)
            r[v] = {"w_bitexact": bool(torch.equal(q.view(torch.uint8), o.view(torch.uint8))),
                    "w_mismatch_frac": float((q.view(torch.uint8) != o.view(torch.uint8)).float().mean()),
                    "scale_bitexact": bool(torch.equal(s, si)), "scale_maxrel": float(((s - si).abs() / si).max())}
        res[n] = r
print(json.dumps(res))
