#!/usr/bin/env python3
"""GDN output chain, fused: FLA gated RMSNorm (norm_before_gate, head dim 128) + per-token-group (128) FP8 quant with
column-major scales, in one Triton kernel. Checks A bytes, scales and the CUTLASS out_proj GEMM against today's
rmsnorm_fn -> per_token_group_quant_fp8 -> cutlass_scaled_mm, and times both. Prints ONE json."""
import json, torch, triton, triton.language as tl
from vllm import _custom_ops as ops
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

FP8_MAX = 448.0


@triton.jit
def _gdn_norm_quant(X, Z, W, Q, S, M, T, NH, eps, qeps,
                    N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, ACTIVATION: tl.constexpr):
    # Norm: FLA layer_norm_fwd_kernel's expressions verbatim (IS_RMS_NORM, HAS_Z, NORM_BEFORE_GATE, no bias).
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * N
    col_offsets = cols[None, :]
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    x = tl.load(X + row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(Z + row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    # FLA stores y in the output dtype (bf16); the quant kernel reads that bf16 value.
    yf = y.to(tl.bfloat16).to(tl.float32)
    # Quant: per_token_group_quant.cu -- absmax = max(eps, |y|), s = absmax / 448, q = clamp(y / s, +-448).
    absmax = tl.maximum(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=1), qeps)
    s = tl.math.div_rn(absmax, tl.full(absmax.shape, 448.0, tl.float32))  # exact: the fast-math '/' gave 16k mismatches
    q = tl.math.div_rn(yf, s[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(Q + row_offsets + col_offsets, q.to(tl.float8e4nv), mask=mask)
    # row r = token * NH + head; column-major scale buffer [NH, T]: index head * T + token.
    tok = rows // NH; head = rows % NH
    tl.store(S + head * T + tok, s, mask=rows < M)


def fused(x, z, w, eps, activation="silu"):
    T, NH, D = x.shape
    M = T * NH
    q = torch.empty(T, NH * D, device=x.device, dtype=torch.float8_e4m3fn)
    sbuf = torch.empty(NH, T, device=x.device, dtype=torch.float32)
    _gdn_norm_quant[(triton.cdiv(M, 4),)](x, z, w, q, sbuf, M, T, NH, eps, 1e-10, N=D, BLOCK_N=128,
                                          ROWS_PER_BLOCK=4, ACTIVATION=activation, num_warps=1)
    return q, sbuf.t()


def today(x, z, w, eps, activation="silu"):
    y = rmsnorm_fn(x, w, None, z=z, eps=eps, group_size=None, norm_before_gate=True, activation=activation)
    return per_token_group_quant_fp8(y.flatten(-2), 128, column_major_scales=True, use_ue8m0=False)


def bench(fn, reps=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


torch.manual_seed(0); dev = "cuda"; NH, D, NOUT = 48, 128, 2560
Bw = (torch.randn(NOUT, NH * D, device=dev) * 0.05).to(torch.float8_e4m3fn)
Bs = (torch.rand(NOUT // 128, NH * D // 128, device=dev) * 0.01 + 0.001)
out = {}
for seed, wdt in ((0, torch.float32), (0, torch.bfloat16), (7, torch.float32), (7, torch.bfloat16)):
    torch.manual_seed(seed)
    w = (1.0 + 0.1 * torch.randn(D, device=dev)).to(wdt)
    for T in (3456, 595, 128):
        x = torch.randn(T, NH, D, device=dev).bfloat16() * 3; z = torch.randn(T, NH, D, device=dev).bfloat16()
        qa, sa = today(x, z, w, 1e-6); qb, sb = fused(x, z, w, 1e-6)
        ga = ops.cutlass_scaled_mm(qa, Bw.t(), out_dtype=torch.bfloat16, scale_a=sa, scale_b=Bs.t())
        gb = ops.cutlass_scaled_mm(qb, Bw.t(), out_dtype=torch.bfloat16, scale_a=sb, scale_b=Bs.t())
        out[f"s{seed}-{str(wdt)[6:]}-{T}"] = {
            "A_bitexact": bool(torch.equal(qa.view(torch.uint8), qb.view(torch.uint8))),
            "A_mismatch": int((qa.view(torch.uint8) != qb.view(torch.uint8)).sum()),
            "scale_bitexact": bool(torch.equal(sa, sb)), "gemm_bitexact": bool(torch.equal(ga, gb)),
            "sa_stride": list(sa.stride()), "sb_stride": list(sb.stride()),
            "today_ms": bench(lambda: today(x, z, w, 1e-6)), "fused_ms": bench(lambda: fused(x, z, w, 1e-6))}
print(json.dumps(out))
