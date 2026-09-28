# SPDX-License-Identifier: Apache-2.0
# FNMOEFUSE (2026-09-28, speed-of-light §5ag): NVFP4 MoE prefill in Triton on SM12x, reading FlashInfer's processed
# tensors in place (w13 = [up | gate] after vLLM's w1w3 -> w3w1 reorder, 128x4-swizzled e4m3 block scales, per-expert
# alphas and global activation scales). Three kernels:
#   gemm1: A rows gathered from the per-token FP4 input (no expandInputRows), gate and up tiles in one CTA, epilogue
#          alpha -> bf16 -> silu(gate)*up -> bf16 -> NVFP4 with FlashInfer's fast-math recipe (no doActivation, no
#          bf16 GEMM1 output round trip);
#   gemm2: alpha2 -> bf16 rows per (token, k);
#   finalize: router-weighted sum over k in fixed order (k = 0..topk-1, fp32), deterministic like prod's DETFIN.
# Native FP4 MMA needs sm_120 codegen (Triton #10010 enables sm_121 only after 3.7.1); moe_fp4_prefill compiles these
# kernels for sm_120 (sm_120 cubins run on sm_121) and leaves every other Triton kernel alone.
import torch

import triton
import triton.language as tl


@triton.jit
def _rcp(x):
    # rcp.approx.ftz.f32, as FlashInfer's reciprocal_approximate_ftz
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _e2m1(x):
    # cvt.rn.satfinite.e2m1 by thresholds: round-to-nearest-even on the grid 0, .5, 1, 1.5, 2, 3, 4, 6, saturating.
    ax = tl.abs(x)
    c = ((ax > 0.25).to(tl.uint8) + (ax >= 0.75).to(tl.uint8) + (ax > 1.25).to(tl.uint8) +
         (ax >= 1.75).to(tl.uint8) + (ax > 2.5).to(tl.uint8) + (ax >= 3.5).to(tl.uint8) + (ax > 5.0).to(tl.uint8))
    sign = (x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)
    return c | (sign << 3)


@triton.jit
def _sw(m, kb, KP4: tl.constexpr):
    # Offset of (row m, scale column kb) in vLLM's swizzle_blockscale layout: [M/128, K/4, 32, 4, 4].
    return (((m // 128) * KP4 + kb // 4) * 32 + m % 32) * 16 + ((m // 32) % 4) * 4 + kb % 4


@triton.jit
def _nvfp4_block(v, g, BM: tl.constexpr, BN: tl.constexpr):
    # v: [BM, BN] fp32 holding bf16 values. FlashInfer cvt_warp_fp16_to_fp4 (fast math): SF = e4m3(g * amax * rcp(6)),
    # scale = rcp(float(SF) * rcp(g)), q = e2m1(v * scale). Returns packed codes [BM, BN/2] and SF [BM, BN/16].
    v3 = tl.reshape(v, [BM, BN // 16, 16])
    amax = tl.max(tl.abs(v3), axis=2)
    six = tl.full([BM, BN // 16], 6.0, tl.float32)
    sf = (g * (amax * _rcp(six))).to(tl.float8e4nv)
    gi = _rcp(tl.full([BM, BN // 16], 1.0, tl.float32) * g)
    scale = tl.where(amax != 0.0, _rcp(sf.to(tl.float32) * gi), 0.0)
    code = _e2m1(v3 * scale[:, :, None])
    c0, c1 = tl.split(tl.reshape(code, [BM, BN // 2, 2]))
    return c0 | (c1 << 4), sf


@triton.jit
def _quant_rows(X, Q, S, M, H: tl.constexpr, g, BM: tl.constexpr, BN: tl.constexpr):
    # Test helper: the same recipe as a standalone row quant (compared bit for bit with ops.scaled_fp4_quant).
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    msk = rm[:, None] < M
    v = tl.load(X + rm[:, None] * H + rn[None, :], mask=msk, other=0.0).to(tl.float32)
    q, sf = _nvfp4_block(v, g, BM, BN)
    tl.store(Q + rm[:, None] * (H // 2) + (tl.program_id(1) * (BN // 2) + tl.arange(0, BN // 2))[None, :], q,
             mask=msk)
    tl.store(S + rm[:, None] * (H // 16) + (tl.program_id(1) * (BN // 16) + tl.arange(0, BN // 16))[None, :], sf,
             mask=msk)


@triton.jit
def _gemm1_swiglu_fp4(XQ, XS, W, WS, SORTED, EXPERTS, NTPP, ALPHA1, G2, OQ, OS, num_valid, NMB,
                      TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid % NMB
    pid_n = pid // NMB
    if pid_m * BM >= tl.load(NTPP):
        return
    e = tl.load(EXPERTS + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm)
    tok = tl.where(sid < num_valid, sid // TOPK, 0)
    rn = pid_n * BN + tl.arange(0, BN)
    rkb = tl.arange(0, BK // 2)
    rks = tl.arange(0, BK // 16)
    wb = W + e * (2 * I * (H // 2))
    wsb = WS + e * (2 * I * (H // 16))
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, H, BK):
        kb = k0 // 2 + rkb
        ks = k0 // 16 + rks
        a = tl.load(XQ + tok[:, None] * (H // 2) + kb[None, :])
        a_s = tl.load(XS + _sw(tok[:, None], ks[None, :], H // 64))
        bu = tl.load(wb + rn[:, None] * (H // 2) + kb[None, :])
        bg = tl.load(wb + (I + rn)[:, None] * (H // 2) + kb[None, :])
        su = tl.load(wsb + _sw(rn[:, None], ks[None, :], H // 64))
        sg = tl.load(wsb + _sw((I + rn)[:, None], ks[None, :], H // 64))
        acc_u = tl.dot_scaled(a, a_s, "e2m1", tl.trans(bu), su, "e2m1", acc_u)
        acc_g = tl.dot_scaled(a, a_s, "e2m1", tl.trans(bg), sg, "e2m1", acc_g)
    alpha = tl.load(ALPHA1 + e)
    up = (acc_u * alpha).to(tl.bfloat16).to(tl.float32)
    gate = (acc_g * alpha).to(tl.bfloat16).to(tl.float32)
    h = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up
    h = h.to(tl.bfloat16).to(tl.float32)
    q, sf = _nvfp4_block(h, tl.load(G2 + e), BM, BN)
    tl.store(OQ + rm[:, None] * (I // 2) + (pid_n * (BN // 2) + tl.arange(0, BN // 2))[None, :], q)
    tl.store(OS + rm[:, None] * (I // 16) + (pid_n * (BN // 16) + tl.arange(0, BN // 16))[None, :], sf)


@triton.jit
def _gemm2(IQ, IS, W, WS, SORTED, EXPERTS, NTPP, ALPHA2, Y, num_valid, NMB,
           H: tl.constexpr, I: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid % NMB
    pid_n = pid // NMB
    if pid_m * BM >= tl.load(NTPP):
        return
    e = tl.load(EXPERTS + pid_m).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    sid = tl.load(SORTED + rm)
    rn = pid_n * BN + tl.arange(0, BN)
    rkb = tl.arange(0, BK // 2)
    rks = tl.arange(0, BK // 16)
    wb = W + e * (H * (I // 2))
    wsb = WS + e * (H * (I // 16))
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, I, BK):
        kb = k0 // 2 + rkb
        ks = k0 // 16 + rks
        a = tl.load(IQ + rm[:, None] * (I // 2) + kb[None, :])
        a_s = tl.load(IS + rm[:, None] * (I // 16) + ks[None, :])
        b = tl.load(wb + rn[:, None] * (I // 2) + kb[None, :])
        s = tl.load(wsb + _sw(rn[:, None], ks[None, :], I // 64))
        acc = tl.dot_scaled(a, a_s, "e2m1", tl.trans(b), s, "e2m1", acc)
    y = (acc * tl.load(ALPHA2 + e)).to(tl.bfloat16)
    tl.store(Y + sid[:, None].to(tl.int64) * H + rn[None, :], y, mask=(sid < num_valid)[:, None])


@triton.jit
def _finalize(Y, TW, OUT, H: tl.constexpr, TOPK: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    rh = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros([BH], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        w = tl.load(TW + t * TOPK + k)
        acc += w * tl.load(Y + (t * TOPK + k) * H + rh).to(tl.float32)
    tl.store(OUT + t * H + rh, acc.to(tl.bfloat16))


_LOGGED = []


def _log_once(m):
    if not _LOGGED:
        _LOGGED.append(m)
        try:
            from vllm.logger import init_logger
            init_logger(__name__).info("FNMOEFUSE Triton NVFP4 prefill MoE ran (M=%d)", m)
        except ImportError:
            pass


CFG1 = dict(BM=64, BN=64, BK=256, num_warps=4, num_stages=3)
CFG2 = dict(BM=64, BN=128, BK=128, num_warps=4, num_stages=3)


def moe_fp4_prefill(xq, xs, w13, w13_s, w2, w2_s, alpha1, a2_gscale, alpha2, topk_ids, topk_weights, out,
                    cfg1=None, cfg2=None, align=None):
    """xq [M, H/2] uint8 + xs (swizzled e4m3, from ops.scaled_fp4_quant); w13 [E, 2I, H/2], w13_s swizzled
    [E, 2I, H/16]; w2 [E, H, I/2], w2_s swizzled [E, H, I/16]; alpha1/a2_gscale/alpha2 fp32 [E]; out [M, H] bf16.
    The kernels compile for sm_120 (native FP4 MMA); Triton keys its in-memory cache without the arch, so the
    override applies to these three kernels only and is restored afterwards."""
    _log_once(out.shape[0])
    prev = triton.knobs.runtime.override_arch
    triton.knobs.runtime.override_arch = "sm120"
    try:
        return _moe_fp4_prefill(xq, xs, w13, w13_s, w2, w2_s, alpha1, a2_gscale, alpha2, topk_ids, topk_weights,
                                out, cfg1, cfg2, align)
    finally:
        triton.knobs.runtime.override_arch = prev


def _moe_fp4_prefill(xq, xs, w13, w13_s, w2, w2_s, alpha1, a2_gscale, alpha2, topk_ids, topk_weights, out,
                     cfg1, cfg2, align):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    c1 = {**CFG1, **(cfg1 or {})}
    c2 = {**CFG2, **(cfg2 or {})}
    assert c1["BM"] == c2["BM"], "gemm1 and gemm2 share the sorted-row blocks"
    M, H = out.shape
    E, I2 = w13.shape[0], w13.shape[1]
    I = I2 // 2
    TOPK = topk_ids.shape[1]
    BM = c1["BM"]
    sorted_ids, expert_ids, ntpp = align if align is not None else moe_align_block_size(topk_ids, BM, E)
    NMB = expert_ids.numel()
    rows = sorted_ids.numel()
    dev = out.device
    iq = torch.empty(rows, I // 2, dtype=torch.uint8, device=dev)
    isf = torch.empty(rows, I // 16, dtype=torch.float8_e4m3fn, device=dev)
    num_valid = M * TOPK
    _gemm1_swiglu_fp4[(NMB * (I // c1["BN"]),)](
        xq, xs, w13, w13_s, sorted_ids, expert_ids, ntpp, alpha1, a2_gscale, iq, isf, num_valid, NMB,
        TOPK=TOPK, H=H, I=I, BM=BM, BN=c1["BN"], BK=c1["BK"], num_warps=c1["num_warps"],
        num_stages=c1["num_stages"])
    y = torch.empty(num_valid, H, dtype=torch.bfloat16, device=dev)
    _gemm2[(NMB * (H // c2["BN"]),)](
        iq, isf, w2, w2_s, sorted_ids, expert_ids, ntpp, alpha2, y, num_valid, NMB,
        H=H, I=I, BM=BM, BN=c2["BN"], BK=c2["BK"], num_warps=c2["num_warps"], num_stages=c2["num_stages"])
    _finalize[(M, H // 512)](y, topk_weights, out, H=H, TOPK=TOPK, BH=512, num_warps=4)
    return out
