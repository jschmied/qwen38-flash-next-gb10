"""Crossover sweep: fused op (FN_HCFUSE_MIN=1, set before import) vs today's kernel sequence. Prints ONE json."""
import os; os.environ["FN_HCFUSE_MIN"] = "1"
import json, torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_silu, _hc_gate_mix
from vllm.models.qwen4_exp.nvidia.ops.hc_fused import hc_combine_mix_fused
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H
torch.manual_seed(2); dev = "cuda"
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()


def today(res, blk, inj):
    o, xn = _hc_combine_norm(res, blk, inj, w, EPS, HC); d = F.linear(xn, wd)
    return o, _hc_gate_mix(xn, F.linear(_hc_silu(d[:, :R], HC), wu), HC), d


def bench(fn, reps=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


out = {}
for M in ():
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    t = bench(lambda: today(res, blk, inj)); f = bench(lambda: hc_combine_mix_fused(res, blk, inj, w, wd, wu, EPS, HC, R))
    out[M] = {"today_ms": t, "fused_ms": f, "delta_pct": round(100 * (f / t - 1), 1)}
print(json.dumps(out))


import triton, triton.language as tl
from vllm.models.qwen4_exp.nvidia.ops.hc_fused import _k1_combine_rrms, _k3_up_mix


@triton.jit
def _k2s_down(out_ptr, rrms_ptr, w_ptr, wd_ptr, p_ptr, M, N, s_out, s_wd,
              KS: tl.constexpr, HC_DIM: tl.constexpr, HC: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # One K slice of xn @ Wd^T (xn rebuilt on load); fp32 partial to p[slice, m, n].
    pid = tl.program_id(0); sl = tl.program_id(1); num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n; pid_n = pid % num_n
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < M; mn = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(sl * KS, sl * KS + KS, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(out_ptr + rm[:, None] * s_out + rk[None, :], mm[:, None], other=0.0).to(tl.float32)
        rr = tl.load(rrms_ptr + rm * HC + k0 // HC_DIM, mm, other=0.0)
        w = tl.load(w_ptr + rk).to(tl.float32)
        y = a * rr[:, None]
        y += y * w[None, :]
        b = tl.load(wd_ptr + rn[None, :] * s_wd + rk[:, None], mn[None, :], other=0.0)
        acc = tl.dot(y.to(tl.bfloat16), b, acc)
    tl.store(p_ptr + (sl * M + rm[:, None]) * N + rn[None, :], acc, mm[:, None] & mn[None, :])


@triton.jit
def _k2r_reduce(p_ptr, c_ptr, M, N, s_c, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, BLOCK); m = cols < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT):  # fixed order: deterministic
        acc += tl.load(p_ptr + (s * M + row) * N + cols, m, other=0.0)
    tl.store(c_ptr + row * s_c + cols, acc.to(tl.bfloat16), m)


def fused_splitk(res, blk, inj, SPLIT, k2=(64, 128, 64, 4, 3), k3=(64, 64, 64, 4, 2)):
    M = res.shape[0]
    out = torch.empty_like(res); rrms = torch.empty(M, HC, device=dev, dtype=torch.float32)
    _k1_combine_rrms[(M * HC,)](blk, res, inj, out, rrms, blk.stride(0), res.stride(0), inj.stride(0), out.stride(0),
                                H, HC, EPS, 512)
    part = torch.empty(SPLIT, M, ND, device=dev, dtype=torch.float32)
    bm, bn, bk, nw, ns = k2
    _k2s_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn), SPLIT)](out, rrms, w, wd, part, M, ND, out.stride(0),
                                                                 wd.stride(0), D // SPLIT, H, HC, bm, bn, bk,
                                                                 num_warps=nw, num_stages=ns)
    d = torch.empty(M, ND, device=dev, dtype=torch.bfloat16)
    _k2r_reduce[(M,)](part, d, M, ND, d.stride(0), SPLIT, 512)
    lora = _hc_silu(d[:, :R], HC)
    y = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    bm, bn, bk, nw, ns = k3
    _k3_up_mix[(triton.cdiv(M, bm), H // bn)](lora, wu, out, rrms, w, y, M, lora.stride(0), wu.stride(0),
                                              out.stride(0), y.stride(0), R, H, HC, bm, bn, bk, num_warps=nw,
                                              num_stages=ns)
    return out, y, d


out2 = {}
for M in (64, 128, 192, 256, 384, 512, 640, 1024, 3456):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    t = bench(lambda: today(res, blk, inj)); f = bench(lambda: hc_combine_mix_fused(res, blk, inj, w, wd, wu, EPS, HC, R))
    o_t, y_t, d_t = today(res, blk, inj)
    r = {"today_ms": t, "fused_ms": f}
    for SPLIT in (4, 8, 16):
        o_s, y_s, d_s = fused_splitk(res, blk, inj, SPLIT)
        o_s2, y_s2, d_s2 = fused_splitk(res, blk, inj, SPLIT)
        r[f"split{SPLIT}_ms"] = bench(lambda: fused_splitk(res, blk, inj, SPLIT))
        r[f"split{SPLIT}_ymax"] = float((y_s.float() - y_t.float()).abs().max())
        r[f"split{SPLIT}_repro"] = bool(torch.equal(y_s, y_s2) and torch.equal(d_s, d_s2))
    out2[M] = r
print(json.dumps(out2))
