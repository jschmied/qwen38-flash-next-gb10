#!/usr/bin/env python3
"""HC-fusion standalone prototype (§5aa item 1). One hyper-connection block (combine_and_mix with injection):
  today: _hc_combine_norm -> F.linear(xn, Wd) -> split -> _hc_silu -> F.linear(lora, Wu) -> _hc_gate_mix
  fused: k1 (residual + rrms only) -> k2 down GEMM normalizing on load -> _hc_silu -> k3 up GEMM + gate-mix epilogue
Checks residual / xn bit-exactness and block-input error vs an fp32 reference; times both at M = 3456 and M = 6.
Prints ONE json."""
import json, torch, triton, triton.language as tl
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_silu, _hc_gate_mix

HC, H, R, EPS = 4, 2560, 320, 1e-6
D, ND = HC * H, 336  # down output: 320 lora + 4 injection + 12 pad


@triton.jit
def _k1_combine_rrms(block_ptr, res_ptr, inj_ptr, out_ptr, rrms_ptr, s_block, s_res, s_inj, s_out,
                     HC_DIM: tl.constexpr, HC: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Same tiles, loads, rounding and reduction as _hc_combine_norm_kernel; stores rrms instead of y.
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)
    pid = tl.program_id(0); row = pid // HC; stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD); mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    res = tl.load(res_ptr + row * s_res + offs, mask_inner, other=0.0)
    inj = tl.load(inj_ptr + row * s_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * s_block + offs_inner, mask_inner, other=0.0)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * s_out + offs, out, mask=mask_inner)
    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    tl.store(rrms_ptr + row * HC + stream, tl.rsqrt(sum_sq / HC_DIM + EPS))


@triton.jit
def _k2_down(out_ptr, rrms_ptr, w_ptr, wd_ptr, c_ptr, M, N, s_out, s_wd, s_c,
             K: tl.constexpr, HC_DIM: tl.constexpr, HC: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # C = xn @ Wd^T with xn rebuilt on load exactly as _hc_combine_norm_kernel writes y (bf16).
    pid = tl.program_id(0); num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n; pid_n = pid % num_n          # N fastest: the programs of one M tile share A in L2
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < M; mn = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(out_ptr + rm[:, None] * s_out + rk[None, :], mm[:, None], other=0.0).to(tl.float32)
        rr = tl.load(rrms_ptr + rm * HC + k0 // HC_DIM, mm, other=0.0)
        w = tl.load(w_ptr + rk).to(tl.float32)
        y = a * rr[:, None]
        y += y * w[None, :]
        b = tl.load(wd_ptr + rn[None, :] * s_wd + rk[:, None], mn[None, :], other=0.0)
        acc = tl.dot(y.to(tl.bfloat16), b, acc)
    tl.store(c_ptr + rm[:, None] * s_c + rn[None, :], acc.to(tl.bfloat16), mm[:, None] & mn[None, :])


@triton.jit
def _mix(acc, g, s: tl.constexpr, rm, mm, ri, out_ptr, rrms_ptr, w_ptr, s_out, HC_DIM: tl.constexpr, HC: tl.constexpr):
    g = g.to(tl.bfloat16).to(tl.float32)
    col = s * HC_DIM + ri
    x = tl.load(out_ptr + rm[:, None] * s_out + col[None, :], mm[:, None], other=0.0).to(tl.float32)
    rr = tl.load(rrms_ptr + rm * HC + s, mm, other=0.0)
    w = tl.load(w_ptr + col).to(tl.float32)
    xn = x * rr[:, None]
    xn += xn * w[None, :]
    xn = xn.to(tl.bfloat16).to(tl.float32)
    return acc + tl.sigmoid(g) * xn


@triton.jit
def _kx_xn(out_ptr, rrms_ptr, w_ptr, xn_ptr, s_out, HC_DIM: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr):
    # xn rebuilt from (out, rrms, w) with the prologue's expression, for the bit-exactness check only.
    row = tl.program_id(0); t = tl.program_id(1)
    k = t * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(out_ptr + row * s_out + k).to(tl.float32) * tl.load(rrms_ptr + row * HC + k // HC_DIM)
    y += y * tl.load(w_ptr + k).to(tl.float32)
    tl.store(xn_ptr + row * s_out + k, y.to(tl.bfloat16))


@triton.jit
def _k3_up_mix(lora_ptr, wu_ptr, out_ptr, rrms_ptr, w_ptr, y_ptr, M, s_lora, s_wu, s_out, s_y,
               R: tl.constexpr, HC_DIM: tl.constexpr, HC: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # y[m, i] = mean_s sigmoid(bf16(lora @ Wu[s*H + i]^T)) * xn[m, s*H + i], streams in order 0..3 as _hc_gate_mix.
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM); ri = pid_n * BN + tl.arange(0, BN)
    mm = rm < M
    a0 = tl.zeros([BM, BN], dtype=tl.float32); a1 = tl.zeros([BM, BN], dtype=tl.float32)
    a2 = tl.zeros([BM, BN], dtype=tl.float32); a3 = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, R, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(lora_ptr + rm[:, None] * s_lora + rk[None, :], mm[:, None], other=0.0)
        a0 = tl.dot(a, tl.load(wu_ptr + (0 * HC_DIM + ri)[None, :] * s_wu + rk[:, None]), a0)
        a1 = tl.dot(a, tl.load(wu_ptr + (1 * HC_DIM + ri)[None, :] * s_wu + rk[:, None]), a1)
        a2 = tl.dot(a, tl.load(wu_ptr + (2 * HC_DIM + ri)[None, :] * s_wu + rk[:, None]), a2)
        a3 = tl.dot(a, tl.load(wu_ptr + (3 * HC_DIM + ri)[None, :] * s_wu + rk[:, None]), a3)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    acc = _mix(acc, a0, 0, rm, mm, ri, out_ptr, rrms_ptr, w_ptr, s_out, HC_DIM, HC)
    acc = _mix(acc, a1, 1, rm, mm, ri, out_ptr, rrms_ptr, w_ptr, s_out, HC_DIM, HC)
    acc = _mix(acc, a2, 2, rm, mm, ri, out_ptr, rrms_ptr, w_ptr, s_out, HC_DIM, HC)
    acc = _mix(acc, a3, 3, rm, mm, ri, out_ptr, rrms_ptr, w_ptr, s_out, HC_DIM, HC)
    acc /= HC
    tl.store(y_ptr + rm[:, None] * s_y + ri[None, :], acc.to(tl.bfloat16), mm[:, None])


def today(res, blk, inj, w, wd, wu):
    out, xn = _hc_combine_norm(res, blk, inj, w, EPS, HC)
    d = F.linear(xn, wd)
    lora, inj2, _ = d.split([R, HC, ND - R - HC], dim=-1)
    gate = F.linear(_hc_silu(lora, HC), wu)
    return out, xn, d, _hc_gate_mix(xn, gate, HC)


def fused(res, blk, inj, w, wd, wu, cfg):
    M = res.shape[0]
    out = torch.empty_like(res); rrms = torch.empty(M, HC, device=res.device, dtype=torch.float32)
    _k1_combine_rrms[(M * HC,)](blk, res, inj, out, rrms, blk.stride(0), res.stride(0), inj.stride(0), out.stride(0),
                                H, HC, EPS, 512)
    d = torch.empty(M, ND, device=res.device, dtype=torch.bfloat16)
    bm, bn, bk, nw, ns = cfg["k2"]
    _k2_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn),)](out, rrms, w, wd, d, M, ND, out.stride(0), wd.stride(0),
                                                          d.stride(0), D, H, HC, bm, bn, bk, num_warps=nw, num_stages=ns)
    lora = _hc_silu(d[:, :R], HC)
    y = torch.empty(M, H, device=res.device, dtype=torch.bfloat16)
    bm, bn, bk, nw, ns = cfg["k3"]
    _k3_up_mix[(triton.cdiv(M, bm), H // bn)](lora, wu, out, rrms, w, y, M, lora.stride(0), wu.stride(0),
                                              out.stride(0), y.stride(0), R, H, HC, bm, bn, bk, num_warps=nw,
                                              num_stages=ns)
    return out, rrms, d, y


def fp32_ref(out, w, wd, wu):
    o = out.float().view(-1, HC, H)
    xn = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)).reshape(out.shape[0], D)
    xn = xn + xn * w.float()
    lora = (xn @ wd.float().t())[:, :R] / HC
    lora = lora * torch.sigmoid(lora)
    gate = lora @ wu.float().t()
    return (torch.sigmoid(gate) * xn).view(-1, HC, H).mean(1)


def bench(fn, reps=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


torch.manual_seed(0); dev = "cuda"
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wd[R + HC:] = 0
wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()
res_out = {}
CFGS = [{"k2": (64, 128, 64, 4, 3), "k3": (64, 64, 64, 4, 2)},
        {"k2": (128, 128, 64, 8, 3), "k3": (64, 128, 64, 8, 2)},
        {"k2": (64, 64, 64, 4, 4), "k3": (32, 64, 64, 4, 2)},
        {"k2": (128, 64, 64, 4, 3), "k3": (128, 64, 32, 8, 2)}]
for M in (3456, 6):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    t_out, t_xn, t_d, t_y = today(res, blk, inj, w, wd, wu)
    r = {"today_ms": bench(lambda: today(res, blk, inj, w, wd, wu))}
    ref = fp32_ref(t_out, w, wd, wu)
    e_t = (t_y.float() - ref).abs()
    r["today_err"] = [float(e_t.max()), float(e_t.mean())]
    for i, cfg in enumerate(CFGS):
        try:
            f_out, f_rr, f_d, f_y = fused(res, blk, inj, w, wd, wu, cfg)
        except Exception as ex:  # a config that does not compile
            r[f"cfg{i}"] = type(ex).__name__ + ": " + str(ex)[:120]; continue
        xn_re = torch.empty_like(f_out); _kx_xn[(M, D // 512)](f_out, f_rr, w, xn_re, f_out.stride(0), H, HC, 512)
        e_f = (f_y.float() - ref).abs()
        r[f"cfg{i}"] = {"ms": bench(lambda: fused(res, blk, inj, w, wd, wu, cfg)),
                        "out_bitexact": bool(torch.equal(f_out, t_out)), "xn_bitexact": bool(torch.equal(xn_re, t_xn)),
                        "err": [float(e_f.max()), float(e_f.mean())],
                        "vs_today_max": float((f_y.float() - t_y.float()).abs().max())}
    res_out[str(M)] = r
print(json.dumps(res_out))
