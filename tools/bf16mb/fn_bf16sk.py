# SPDX-License-Identifier: Apache-2.0
"""FNBF16SK: deterministic split-K Triton GEMM for small-M BF16 linears (jschmied 2026-09-25, local, not upstream).

cuBLAS serves the Flash-Next BF16 linears at M <= 16 with the sm80 `cutlass_80_wmma` 32-thread-block kernels:
hyper-connection mixer down [324x10240] 40.5 us in-model vs a 30.2 us byte floor, GDN in_proj_ba [96x2560] 17.2 vs
2.2 (speed-of-light step 3). Two passes, no atomics: pass 1 writes fp32 partials [S, M, N], pass 2 sums them in a
fixed order, so the result is bitwise reproducible. Dispatch happens inside a custom op, at run time, so a
torch.compile trace cannot freeze the M branch.
FN_BF16SK=1 turns it on; UnquantizedLinearMethod.apply routes weights whose (N, K) is in CONFIGS here.
"""
import torch
import triton
import triton.language as tl

MAX_M = 16
# (N, K) -> (BN, BK, SPLIT); chosen by tools/bf16mb/bf16sk_bench.py
CONFIGS = {  # (N, K): {M bucket: (BN, BK, SPLIT)}, from notes/data/bf16sk-0925.json
    (324, 10240): {1: (16, 256, 4), 4: (16, 256, 2), 16: (16, 256, 2)},      # hyper-connection mixer down+inject
    (336, 10240): {1: (16, 256, 4), 4: (16, 256, 2), 16: (16, 256, 2)},      # same, padded to 16 rows (nvidia path)
    (10240, 320): {1: (16, 128, 1), 4: (16, 128, 1), 16: (16, 256, 1)},      # hyper-connection mixer up
    (96, 2560): {1: (16, 256, 8), 4: (16, 256, 8), 16: (16, 256, 8)},        # GDN in_proj_ba
    (512, 2560): {1: (16, 256, 1), 4: (16, 256, 1), 16: (16, 256, 1)},       # router
    (1280, 2560): {1: (32, 256, 1), 4: (64, 256, 1), 16: (32, 256, 1)},      # shared expert gate_up
    (2560, 640): {1: (16, 128, 1), 4: (16, 128, 1), 16: (32, 128, 1)},       # shared expert down
}


@triton.jit
def _partial(x_ptr, w_ptr, p_ptr, M, N, K, sxm, KC: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MR: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN); m = tl.arange(0, MR)
    acc = tl.zeros((MR, BN), tl.float32)
    for k0 in range(pk * KC, pk * KC + KC, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=(n[:, None] < N) & (k[None, :] < K), other=0.)
        x = tl.load(x_ptr + m[:, None] * sxm + k[None, :], mask=(m[:, None] < M) & (k[None, :] < K), other=0.)
        acc += tl.dot(x, tl.trans(w))
    tl.store(p_ptr + pk * MR * N + m[:, None] * N + n[None, :], acc, mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(p_ptr, o_ptr, M, N, S: tl.constexpr, MR: tl.constexpr, BN: tl.constexpr):
    pn = tl.program_id(0)
    n = pn * BN + tl.arange(0, BN); m = tl.arange(0, MR)
    msk = (m[:, None] < M) & (n[None, :] < N)
    acc = tl.zeros((MR, BN), tl.float32)
    for s in tl.static_range(S):                                  # fixed order: deterministic
        acc += tl.load(p_ptr + s * MR * N + m[:, None] * N + n[None, :], mask=msk, other=0.)
    tl.store(o_ptr + m[:, None] * N + n[None, :], acc.to(o_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _direct(x_ptr, w_ptr, o_ptr, M, N, K, sxm, BN: tl.constexpr, BK: tl.constexpr, MR: tl.constexpr):
    pn = tl.program_id(0)
    n = pn * BN + tl.arange(0, BN); m = tl.arange(0, MR)
    acc = tl.zeros((MR, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=(n[:, None] < N) & (k[None, :] < K), other=0.)
        x = tl.load(x_ptr + m[:, None] * sxm + k[None, :], mask=(m[:, None] < M) & (k[None, :] < K), other=0.)
        acc += tl.dot(x, tl.trans(w))
    tl.store(o_ptr + m[:, None] * N + n[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=(m[:, None] < M) & (n[None, :] < N))


def bf16sk(x: torch.Tensor, w: torch.Tensor, bn: int, bk: int, split: int) -> torch.Tensor:
    """x [M, K] bf16, unit column stride (M <= 16), w [N, K] bf16 contiguous -> [M, N] bf16."""
    M, K = x.shape; N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    if split == 1:
        _direct[(triton.cdiv(N, bn),)](x, w, out, M, N, K, x.stride(0), BN=bn, BK=bk, MR=MAX_M)
        return out
    kc = triton.cdiv(triton.cdiv(K, split), bk) * bk
    s = triton.cdiv(K, kc)
    part = torch.empty(s, MAX_M, N, device=x.device, dtype=torch.float32)
    _partial[(triton.cdiv(N, bn), s)](x, w, part, M, N, K, x.stride(0), KC=kc, BN=bn, BK=bk, MR=MAX_M)
    _reduce[(triton.cdiv(N, 128),)](part, out, M, N, S=s, MR=MAX_M, BN=128)
    return out


_LOGGED = set()


def _fn_bf16sk_impl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M = x.shape[0]
    key = (M, w.shape[0], w.shape[1])
    if key not in _LOGGED and len(_LOGGED) < 64:
        _LOGGED.add(key)
        from vllm.logger import init_logger
        init_logger(__name__).warning("FNBF16SK call M=%d N=%d K=%d -> %s", M, w.shape[0], w.shape[1],
                                      "triton" if 0 < M <= MAX_M else "F.linear")
    cfg = CONFIGS.get((w.shape[0], w.shape[1]))
    if (cfg is None or M == 0 or M > MAX_M or x.dim() != 2 or x.stride(1) != 1 or not w.is_contiguous()
            or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16):
        return torch.nn.functional.linear(x, w)
    return bf16sk(x, w, *cfg[1 if M <= 1 else 4 if M <= 4 else 16])


def _fn_bf16sk_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


def register():
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="fn_bf16sk", op_func=_fn_bf16sk_impl, fake_impl=_fn_bf16sk_fake)
