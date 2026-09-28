# SPDX-License-Identifier: Apache-2.0
# FNHCFUSE (2026-09-28, speed-of-light §5aa): fused hyper-connection combine_and_mix for prefill-sized batches.
# Today: _hc_combine_norm (writes residual + xn) -> down GEMM -> hc_silu -> up GEMM (writes gate) -> _hc_gate_mix.
# Fused: K1 writes residual + rrms only; K2 down GEMM rebuilds xn on its A-load; K3 up GEMM applies the gate mix in
# its epilogue. Removes the xn and gate round trips (531 -> ~320 MB per block at 3,456 tokens; -0.92 ms standalone).
# Below FN_HCFUSE_MIN tokens (decode, small prefills) the op runs today's exact kernel sequence. The size branch lives
# inside this custom op so torch.compile cannot freeze it. Env-gated by FN_HCFUSE=1 in hyperconnection.py.
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .hc import _hc_combine_norm, _hc_gate_mix, _hc_silu

logger = init_logger(__name__)
_MIN_M = int(os.environ.get("FN_HCFUSE_MIN", "128"))  # split-K sweep 2026-09-28: fused wins from 128 on


def _split_for(M: int) -> int:
    # Down-GEMM K split by batch size (sweep 2026-09-28): 8 below 192, 4 up to 2047, none above.
    return 8 if M < 192 else (4 if M < 2048 else 1)
_K2 = (64, 128, 64, 4, 3)
_K3 = (64, 64, 64, 4, 2)


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


def _combine_mix(residual: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
                 norm_weight: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, eps: float, hc_count: int,
                 lora_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, D = residual.shape
    H = D // hc_count
    if M < _MIN_M:
        out, xn = _hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count)
        d = F.linear(xn, w_down)
        gate = F.linear(_hc_silu(d[:, :lora_rank], hc_count), w_up)
        return out, _hc_gate_mix(xn, gate, hc_count), d
    logger.info_once("FNHCFUSE fused hyper-connection kernels ran (M=%d, min=%d)", M, _MIN_M)
    ND = w_down.shape[0]
    out = torch.empty_like(residual)
    rrms = torch.empty(M, hc_count, device=residual.device, dtype=torch.float32)
    _k1_combine_rrms[(M * hc_count,)](block_output, residual, injection_logits, out, rrms, block_output.stride(0),
                                      residual.stride(0), injection_logits.stride(0), out.stride(0), H, hc_count,
                                      eps, 512)
    d = torch.empty(M, ND, device=residual.device, dtype=residual.dtype)
    bm, bn, bk, nw, ns = _K2
    split = _split_for(M)
    if split == 1:
        _k2_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn),)](out, rrms, norm_weight, w_down, d, M, ND, out.stride(0),
                                                              w_down.stride(0), d.stride(0), D, H, hc_count, bm, bn,
                                                              bk, num_warps=nw, num_stages=ns)
    else:  # deterministic split-K: fp32 partials, fixed-order reduce
        part = torch.empty(split, M, ND, device=residual.device, dtype=torch.float32)
        _k2s_down[(triton.cdiv(M, bm) * triton.cdiv(ND, bn), split)](out, rrms, norm_weight, w_down, part, M, ND,
                                                                     out.stride(0), w_down.stride(0), D // split, H,
                                                                     hc_count, bm, bn, bk, num_warps=nw,
                                                                     num_stages=ns)
        _k2r_reduce[(M,)](part, d, M, ND, d.stride(0), split, 512)
    lora = _hc_silu(d[:, :lora_rank], hc_count)
    y = torch.empty(M, H, device=residual.device, dtype=residual.dtype)
    bm, bn, bk, nw, ns = _K3
    _k3_up_mix[(triton.cdiv(M, bm), H // bn)](lora, w_up, out, rrms, norm_weight, y, M, lora.stride(0),
                                              w_up.stride(0), out.stride(0), y.stride(0), lora_rank, H, hc_count,
                                              bm, bn, bk, num_warps=nw, num_stages=ns)
    return out, y, d


def _combine_mix_fake(residual, block_output, injection_logits, norm_weight, w_down, w_up, eps, hc_count, lora_rank):
    M, D = residual.shape
    return (residual.new_empty(residual.shape), residual.new_empty(M, D // hc_count),
            residual.new_empty(M, w_down.shape[0]))


direct_register_custom_op(op_name="qwen4_exp_hc_combine_mix_fused", op_func=_combine_mix, fake_impl=_combine_mix_fake)


def hc_combine_mix_fused(residual, block_output, injection_logits, norm_weight, w_down, w_up, eps, hc_count,
                         lora_rank):
    return torch.ops.vllm.qwen4_exp_hc_combine_mix_fused(residual, block_output, injection_logits, norm_weight,
                                                         w_down, w_up, eps, hc_count, lora_rank)
