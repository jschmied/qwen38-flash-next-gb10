#!/usr/bin/env python3
"""Is a GEMM's row 0 the same for M=1 (single-token decode) and larger M (block verification)?

ROCm/AMD counterpart to gemm_m_invariance.py. Same question, same output format, different backends.

--backend: this script selects paths by what the ROCm stack actually runs, not by a flag. The FP8
CUTLASS paths in the sm_120 script (cutlass_scaled_mm, per-channel and blockwise) have no ROCm
equivalent reachable from vLLM here, so they are omitted rather than faked. What is measured instead:

  * BF16 dense, via torch matmul -> hipBLASLt (falling back to rocBLAS). On AWQ/compressed-tensors
    W4A16 checkpoints this is not a corner case: the quantizer leaves self_attn.* and the linear-
    attention (KDA/GDN) projections in BF16, so these shapes run on every decode step.
  * int4 W4A16 fused MoE, via the Triton fused_experts path. This is the routed-expert path for
    GLM-5.3-Flash-class models on ROCm.

Reference measurement, GLM-5.3-Flash-AWQ-W4A16 on 2x AMD Radeon 8060S (gfx1151, RDNA3.5, Strix Halo),
ROCm 10.0 / torch 2.11 / triton 3.8, TP=2, shapes per rank:

  bf16 kda in_proj 12576x4096   row0 == M=1 for M = 1..64, differs from M=128    max|diff| 5.0e-01
  bf16 kda o_proj   4096x4096   row0 == M=1 for M = 1..64, differs from M=128    max|diff| 1.25e-01
  bf16 mla q_b      3072x4096   row0 == M=1 for M = 1..32, differs from M=64     max|diff| 5.0e-01
  int4 moe E=288 N=1024 K=4096  row0 == M=1 at every M tested (1..256)           max|diff| 0

i.e. ROCm BF16 is M-invariant across the whole decode and verification range and only switches
kernel at M >= 128, unlike the sm_120 BF16 cuBLAS row which differs from M=2 on. The int4 MoE was
bit-identical at every M even with per-M tuned tiles that change BLOCK_SIZE_K and SPLIT_K.

CAVEAT for the MoE row: the tuned-config lookup reads VLLM_TUNED_CONFIG_FOLDER. In a standalone
process that variable is usually unset, so the kernel silently uses the stock config and the result
says nothing about a deployment's tuned tiles. Confirm the log line

    Using configuration from .../E=<E>,N=<N>,device_name=<gpu>,dtype=int4_w4a16.json

before trusting it, or run with VLLM_TUNED_CONFIG_FOLDER pointing at the deployed configs.

Usage: gemm_m_invariance_rocm.py   (any vLLM venv with ROCm torch)
"""
import torch

import vllm  # noqa: F401  (registers custom ops)

dev = "cuda"
torch.manual_seed(0)

# Per-rank shapes at TP=2 for GLM-5.3-Flash. Substitute your own; (name, N, K).
SHAPES = [
    ("kda in_proj 12576x4096", 12576, 4096),
    ("kda o_proj 4096x4096", 4096, 4096),
    ("mla q_b 3072x4096", 3072, 4096),
]
MS = [1, 2, 3, 4, 8, 9, 16, 32, 64, 128, 256, 1024, 4096]

# MoE: (num_experts, intermediate_per_rank, hidden, topk, group). M is capped lower because the
# expert weights dominate memory at E=288.
MOE = (288, 1024, 4096, 8, 128)
MS_MOE = [1, 2, 3, 4, 8, 9, 16, 32, 64, 256]


def bf16_hipblaslt(n, k):
    a = torch.randn(max(MS), k, device=dev, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)

    def run(m):
        return a[:m] @ w.t()

    return run


def int4_w4a16_moe(e, n_int, kk, topk, group):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_op

    w1_q = torch.randint(0, 255, (e, 2 * n_int, kk // 2), dtype=torch.uint8, device=dev)
    w2_q = torch.randint(0, 255, (e, kk, n_int // 2), dtype=torch.uint8, device=dev)
    w1_s = torch.rand(e, 2 * n_int, kk // group, dtype=torch.bfloat16, device=dev) * .02 + .01
    w2_s = torch.rand(e, kk, n_int // group, dtype=torch.bfloat16, device=dev) * .02 + .01

    g = torch.Generator(device=dev).manual_seed(11)
    m_max = max(MS_MOE)
    x = torch.randn(m_max, kk, dtype=torch.bfloat16, device=dev, generator=g)
    # Row 0 keeps the same experts and router weights at every M, so row0 is comparable.
    ids = torch.stack([torch.randperm(e, device=dev, generator=g)[:topk]
                       for _ in range(m_max)]).to(torch.int32)
    wts = torch.softmax(torch.randn(m_max, topk, device=dev, dtype=torch.float32,
                                    generator=g), -1)

    def run(m):
        return fused_experts_op(x[:m], w1_q, w2_q, wts[:m], ids[:m], use_int4_w4a16=True,
                                w1_scale=w1_s, w2_scale=w2_s, block_shape=[0, group],
                                global_num_experts=e)

    return run


def report(name, path, run, ms):
    ref = run(1)[0].clone()
    same, diffs = [], []
    for m in ms:
        r0 = run(m)[0]
        same.append("=" if bool(torch.equal(r0, ref)) else "x")
        diffs.append(float((r0.float() - ref.float()).abs().max()))
    line = " ".join(f"{m}:{s}" for m, s in zip(ms, same))
    print(f"{name:24s} {path:18s} row0 vs M=1: {line}   max|diff| {max(diffs):.3e}",
          flush=True)


for name, n, k in SHAPES:
    try:
        report(name, "bf16 hipBLASLt", bf16_hipblaslt(n, k), MS)
    except Exception as ex:
        print(f"{name:24s} {'bf16 hipBLASLt':18s} ERR {str(ex).splitlines()[0][:110]}",
              flush=True)

e, n_int, kk, topk, group = MOE
try:
    report(f"moe E={e} N={n_int} K={kk}", "int4 w4a16 triton",
           int4_w4a16_moe(e, n_int, kk, topk, group), MS_MOE)
except Exception as ex:
    print(f"{'moe int4':24s} {'int4 w4a16 triton':18s} ERR "
          f"{str(ex).splitlines()[0][:110]}", flush=True)
