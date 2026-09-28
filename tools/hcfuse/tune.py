"""Tile-config sweep for the fused HC op's K2 (down, with split-K) and K3 (up + gate mix). Prints ONE json with, per M,
the current config's time, the best config and time for each kernel, and how many configs passed the checks."""
import os; os.environ["FN_HCFUSE_MIN"] = "1"
import itertools, json, torch, triton
from vllm.models.qwen4_exp.nvidia.ops import hc_fused as hf
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_silu
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H; dev = "cuda"; torch.manual_seed(5)
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()
def t(fn, reps=25):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]
out = {}
for M in (3456, 1024, 512, 256, 128):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16(); inj = torch.randn(M, HC, device=dev).bfloat16()
    o = torch.empty_like(res); rr = torch.empty(M, HC, device=dev)
    hf._k1_combine_rrms[(M * HC,)](blk, res, inj, o, rr, blk.stride(0), res.stride(0), inj.stride(0), o.stride(0), H, HC, EPS, 512)
    # reference d / y from the op's current path
    _, yref, dref = hf._combine_mix(res, blk, inj, w, wd, wu, EPS, HC, R)
    d = torch.empty(M, ND, device=dev, dtype=torch.bfloat16)
    def k2(cfg):
        bm, bn, bk, nw, ns, sp = cfg
        if sp == 1:
            hf._k2_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn),)](o, rr, w, wd, d, M, ND, o.stride(0), wd.stride(0), d.stride(0), D, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
        else:
            part = k2.part.setdefault(sp, torch.empty(sp, M, ND, device=dev))
            hf._k2s_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn), sp)](o, rr, w, wd, part, M, ND, o.stride(0), wd.stride(0), D // sp, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
            hf._k2r_reduce[(M,)](part, d, M, ND, d.stride(0), sp, 512)
    k2.part = {}
    r = {"k2": {}, "k3": {}}
    cur2 = (*hf._K2, hf._split_for(M)); r["k2"]["current"] = [list(cur2), round(t(lambda: k2(cur2)), 4)]
    best, ok = None, 0
    for cfg in itertools.product((32, 64, 128), (64, 128), (64, 128), (4, 8), (3, 4), (1, 2, 4, 8)):
        if cfg[2] * cfg[5] > D or (D // cfg[5]) % cfg[2]: continue
        try:
            k2(cfg); torch.cuda.synchronize(); d1 = d.clone(); k2(cfg); torch.cuda.synchronize()
            if not torch.equal(d1, d) or (d1.float() - dref.float()).abs().max() > 0.0625: continue
            ms = t(lambda: k2(cfg)); ok += 1
            if best is None or ms < best[1]: best = (list(cfg), round(ms, 4))
        except Exception:
            continue
    r["k2"]["best"] = best; r["k2"]["passed"] = ok
    k2(cur2); lora = _hc_silu(d[:, :R], HC); y = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    def k3(cfg):
        bm, bn, bk, nw, ns = cfg
        hf._k3_up_mix[(triton.cdiv(M, bm), H // bn)](lora, wu, o, rr, w, y, M, lora.stride(0), wu.stride(0), o.stride(0), y.stride(0), R, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
    cur3 = hf._K3; r["k3"]["current"] = [list(cur3), round(t(lambda: k3(cur3)), 4)]
    k3(cur3); yc = y.clone()
    best, ok = None, 0
    for cfg in itertools.product((32, 64, 128), (32, 64, 128), (32, 64), (4, 8), (2, 3)):
        if R % cfg[2]: continue
        try:
            k3(cfg); torch.cuda.synchronize(); y1 = y.clone(); k3(cfg); torch.cuda.synchronize()
            if not torch.equal(y1, y) or (y1.float() - yc.float()).abs().max() > 0.0625: continue
            ms = t(lambda: k3(cfg)); ok += 1
            if best is None or ms < best[1]: best = (list(cfg), round(ms, 4))
        except Exception:
            continue
    r["k3"]["best"] = best; r["k3"]["passed"] = ok
    out[str(M)] = r
print(json.dumps(out))
