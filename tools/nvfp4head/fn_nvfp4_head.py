# SPDX-License-Identifier: Apache-2.0
"""NVFP4 draft-head slice for Qwen4Exp MTP (jschmied 2026-09-24, local experiment, not upstream).

The drafter's greedy head reads the 32k-row draft-vocab slice every draft step. As BF16 that is 160 MiB; as NVFP4
(E2M1 values, one E4M3 scale per 16 values along K, one FP32 global scale) it is 40 + 5 MiB. The slice only picks
DRAFT tokens, which the target verifies, so quantization can cost acceptance but never changes the output.

Layout (linear, not the swizzled CUTLASS layout): q uint8 [N, K/2] with element 2j in the low nibble and 2j+1 in the
high nibble; s float8_e4m3fn [N, K/16]; g a python float. value = e2m1(q) * s * g.
"""
import torch

from vllm.triton_utils import tl, triton

_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4_rows(w: torch.Tensor):
    """w: float32 [N, K], K % 16 == 0. Plain per-16 max scaling. Returns (q uint8 [N, K/2], s fp8 [N, K/16], g)."""
    N, K = w.shape
    assert K % 32 == 0
    g = float(w.abs().amax()) / (448.0 * 6.0)
    if g == 0.0:
        g = 1.0
    grp = w.view(N, K // 16, 16)
    amax = grp.abs().amax(-1)                                         # [N, K/16]
    s = (amax / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = s.to(torch.float32) * g                                      # effective group scale
    x = torch.where(sf[..., None] > 0, grp / sf[..., None].clamp(min=1e-30), torch.zeros_like(grp))
    mag = x.abs().clamp(max=6.0)
    code = torch.bucketize(mag, torch.tensor(_E2M1_MID, device=w.device, dtype=mag.dtype))  # 0..7
    nib = (code | ((x < 0).to(code.dtype) << 3)).to(torch.uint8).view(N, K)
    q = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return q, s.contiguous(), g


def dequantize_nvfp4_rows(q: torch.Tensor, s: torch.Tensor, g: float) -> torch.Tensor:
    """Reference dequant to float32 [N, K]."""
    N = q.shape[0]
    lo, hi = q & 0xF, q >> 4
    nib = torch.stack((lo, hi), dim=-1).view(N, -1).to(torch.int64)
    table = torch.tensor(_E2M1_MAG, device=q.device, dtype=torch.float32)
    val = table[nib & 7] * torch.where((nib >> 3) == 1, -1.0, 1.0)
    K = val.shape[1]
    return (val.view(N, K // 16, 16) * s.to(torch.float32)[..., None] * g).view(N, K)


@triton.jit
def _nvfp4_rows_gemv_kernel(x_ptr, q_ptr, s_ptr, out_ptr, g, M, N, K,
                            sxm, sqn, ssn, som,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_b = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        b = tl.load(q_ptr + offs_n[:, None] * sqn + offs_b[None, :], mask=nmask[:, None], other=0)
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        nib = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
        e = (nib >> 1) & 3
        m = (nib & 1).to(tl.float32)
        mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
        val = tl.where(((nib >> 3) & 1) == 1, -mag, mag)
        offs_gk = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        sc = tl.load(s_ptr + offs_n[:, None] * ssn + offs_gk[None, :], mask=nmask[:, None], other=0.0)
        sc = sc.to(tl.float32)
        # e2m1 (<=2 mantissa bits) x e4m3 (3 mantissa bits) is exact in bf16 (7), so this cast loses nothing
        w = tl.reshape(tl.reshape(val, (BLOCK_N, BLOCK_K // 16, 16)) * sc[:, :, None], (BLOCK_N, BLOCK_K))
        w = w.to(tl.bfloat16)
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :], mask=mmask[:, None], other=0.0)
        acc += tl.dot(x.to(tl.bfloat16), tl.trans(w))
    acc = acc * g
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :], acc, mask=mmask[:, None] & nmask[None, :])


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        import os
        bn, nw = (os.environ.get("FN_NVFP4_CFG", "64,4").split(",") + ["4"])[:2]
        _CFG = (int(bn), int(nw))
    return _CFG


def nvfp4_rows_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, g: float,
                    block_n: int | None = None, block_k: int = 128, num_warps: int | None = None) -> torch.Tensor:
    """x bf16 [M, K] -> float32 logits [M, N]. Tile config from FN_NVFP4_CFG='block_n,num_warps' (default 64,4)."""
    if block_n is None or num_warps is None:
        block_n, num_warps = _cfg()
    x = x.contiguous()
    M, K = x.shape
    N = q.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, 16))
    _nvfp4_rows_gemv_kernel[grid](x, q, s, out, g, M, N, K,
                                  x.stride(0), q.stride(0), s.stride(0), out.stride(0),
                                  BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps)
    return out
