"""Per-kernel times of the fused HC op at M=3456 (one pass), with the bytes each moves: K1, K2 (split or not), silu, K3."""
import os; os.environ["FN_HCFUSE_MIN"] = "1"
import json, torch, triton
from vllm.models.qwen4_exp.nvidia.ops import hc_fused as hf
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_silu
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H; dev = "cuda"; torch.manual_seed(4)
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()
M = 3456; res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16(); inj = torch.randn(M, HC, device=dev).bfloat16()
out = torch.empty_like(res); rr = torch.empty(M, HC, device=dev); d = torch.empty(M, ND, device=dev, dtype=torch.bfloat16); y = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
def k1(): hf._k1_combine_rrms[(M * HC,)](blk, res, inj, out, rr, blk.stride(0), res.stride(0), inj.stride(0), out.stride(0), H, HC, EPS, 512)
def k2():
    bm, bn, bk, nw, ns = hf._K2
    hf._k2_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn),)](out, rr, w, wd, d, M, ND, out.stride(0), wd.stride(0), d.stride(0), D, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
lora = _hc_silu(d[:, :R], HC)
def k3():
    bm, bn, bk, nw, ns = hf._K3
    hf._k3_up_mix[(triton.cdiv(M, bm), H // bn)](lora, wu, out, rr, w, y, M, lora.stride(0), wu.stride(0), out.stride(0), y.stride(0), R, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
def silu(): _hc_silu(d[:, :R], HC)
def t(fn, reps=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]
MB = {"k1": (M*D*2*2 + M*H*2)/1e6, "k2": (M*D*2 + ND*D*2)/1e6, "silu": M*R*4/1e6, "k3": (M*D*2 + M*H*2 + D*R*2)/1e6}
r = {}
for n, fn in (("k1", k1), ("k2", k2), ("silu", silu), ("k3", k3)):
    ms = t(fn); r[n] = {"ms": round(ms, 4), "MB": round(MB[n], 1), "GBps": round(MB[n] / ms, 1)}
print(json.dumps(r))
