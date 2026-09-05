#!/usr/bin/env python3
"""Is the FP8 GEMM's row 0 the same for M=1 (single-token decode) and M=8 (block verification)?

v2 (2026-09-05): blockwise scale layouts fixed to the production ones — v1's blockwise row was an artifact.

Issue vllm#54928 (Windless84, 2026-09-04): on a dense Qwen3.8-27B-FP8 the verifier's block-shaped forward ranks a
different token than the single-token forward at near-tie positions (E == V != A), eager or compiled. The cheapest
place for such a batch-shape dependence is the GEMM: the sm120 CUTLASS dispatch picks different tile shapes / a
swap-AB kernel by M, and cuBLAS picks by M as well. This script runs the same first row through M = 1..4096 and
reports whether row 0 is bit-identical to the M=1 result, for per-channel FP8 (cutlass_scaled_mm), blockwise FP8
(128x128 scales, the Flash-Next/27B dense path) and plain BF16 cuBLAS, on the model's dense projection shapes.
Usage: gemm_m_invariance.py   (any vLLM venv with _C built for the GPU)"""
import torch
import vllm  # noqa
from vllm import _custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
SHAPES = [("q_proj 12288x2560", 12288, 2560), ("o_proj 5120x5120", 5120, 5120), ("mlp 16384x2560", 16384, 2560)]
MS = [1, 2, 3, 4, 8, 9, 16, 32, 64, 128, 256, 1024, 4096]
FP8 = torch.float8_e4m3fn


def per_channel(n, k):
    a = (torch.randn(max(MS), k, device=dev) * 0.5).to(FP8)
    sa = (torch.rand(max(MS), 1, device=dev) * 0.02 + 0.01).float()
    w = (torch.randn(n, k, device=dev) * 0.5).to(FP8)
    sb = (torch.rand(1, n, device=dev) * 0.02 + 0.01).float()
    b = w.t()  # [K, N] column-major, as vLLM stores it
    def run(m):
        return ops.cutlass_scaled_mm(a[:m], b, sa[:m], sb, torch.bfloat16)
    return run


def blockwise(n, k):
    """Scale layouts as the production caller builds them (CutlassFp8BlockScaledMMKernel):
    activation scales [m, K/128] column-major (M-major), weight scales built as
    [N/128, K/128] and passed transposed (K-major). The sm120 CUTLASS blockwise kernel
    deduces the layouts from the shapes and reads no strides, so row-major scales here
    silently compute a different GEMM whose row 0 depends on m (jahnclawdmonet, #54521,
    2026-09-05): the first version of this script had exactly that defect."""
    a = (torch.randn(max(MS), k, device=dev) * 0.5).to(FP8)
    sa_rows = (torch.rand(max(MS), k // 128, device=dev) * 0.02 + 0.01).float()
    w = (torch.randn(n, k, device=dev) * 0.5).to(FP8)
    sb_nk = (torch.rand(n // 128, k // 128, device=dev) * 0.02 + 0.01).float()
    b = w.t()
    def run(m):
        sa = sa_rows[:m].t().contiguous().t()  # column-major [m, K/128]
        return ops.cutlass_scaled_mm(a[:m], b, sa, sb_nk.t(), torch.bfloat16)
    return run


def bf16_cublas(n, k):
    a = torch.randn(max(MS), k, device=dev).to(torch.bfloat16)
    w = torch.randn(n, k, device=dev).to(torch.bfloat16)
    def run(m):
        return a[:m] @ w.t()
    return run


for name, n, k in SHAPES:
    for path, mk in (("per-channel FP8", per_channel), ("blockwise FP8", blockwise), ("bf16 cuBLAS", bf16_cublas)):
        try:
            run = mk(n, k)
            ref = run(1)[0].clone()
            same, diffs = [], []
            for m in MS:
                r0 = run(m)[0]
                eq = bool(torch.equal(r0, ref))
                same.append("=" if eq else "x")
                diffs.append(float((r0.float() - ref.float()).abs().max()))
            worst = max(diffs)
            line = " ".join(f"{m}:{s}" for m, s in zip(MS, same))
            print(f"{name:20s} {path:16s} row0 vs M=1: {line}   max|diff| {worst:.3e}", flush=True)
        except Exception as ex:
            print(f"{name:20s} {path:16s} ERR {str(ex).splitlines()[0][:120]}", flush=True)
