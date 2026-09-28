#!/usr/bin/env python3
"""FNMOEFUSE standalone (HYPOTHESIS.md). argv: check | bench [M ...]. One MoE layer at the model's shapes (E 512, top-10,
H 2,560, I 640), random NVFP4 weights processed the way vLLM prepares them for FLASHINFER_CUTLASS (swizzled e4m3 block
scales, [up | gate] rows), random routing. check: (a) the Triton quant recipe vs ops.scaled_fp4_quant bit for bit,
(b) Triton prefill MoE vs FlashInfer cutlass_fused_moe (non-fused finalize, as prod's DETFIN) on the same inputs,
(c) run-to-run bit stability. bench: medians of both, and of each Triton stage. Prints ONE json."""
import json, os, sys
os.environ.setdefault("TRITON_OVERRIDE_ARCH", "sm120")
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.autotuner import autotune
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_fp4 as mf

E, TOPK, H, I = 512, 10, 2560, 640
dev = "cuda"
torch.manual_seed(0)
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8)
w13_s = swizzle_blockscale((torch.rand(E, 2 * I, H // 16, device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn))
w2_s = swizzle_blockscale((torch.rand(E, H, I // 16, device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn))
a1g_val = 448.0 * 6.0 / 5.0
a1g = torch.full((E,), a1g_val, device=dev)
alpha1 = torch.full((E,), 1.0 / (a1g_val * 60.0), device=dev)
a2g = torch.full((E,), 448.0 * 6.0 / 20.0, device=dev)
alpha2 = torch.full((E,), 1.0 / (448.0 * 6.0 / 20.0 * 25.0), device=dev)


def inputs(M, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, H, device=dev, generator=g).bfloat16()
    ids = torch.argsort(torch.rand(M, E, device=dev, generator=g), -1)[:, :TOPK].to(torch.int32).contiguous()
    wts = torch.softmax(torch.randn(M, TOPK, device=dev, generator=g), -1).contiguous()
    xq, xs = ops.scaled_fp4_quant(x, a1g[:1], is_sf_swizzled_layout=True)
    return x, ids, wts, xq, xs


def run_fi(xq, xs, ids, wts, M):
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    cutlass_fused_moe(input=xq, token_selected_experts=ids, token_final_scales=wts,
                      fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                      output_dtype=torch.bfloat16, output=out,
                      quant_scales=[a1g, w13_s.view(torch.int32), alpha1, a2g, w2_s.view(torch.int32), alpha2],
                      input_sf=xs, use_fused_finalize=False)
    return out


def run_tr(xq, xs, ids, wts, M, align=None, cfg1=None, cfg2=None, poison=False):
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    if poison:  # torch.empty can return the block FlashInfer's output was just freed from
        out.fill_(float("nan"))
    return mf.moe_fp4_prefill(xq, xs, w13, w13_s, w2, w2_s, alpha1, a2g, alpha2, ids, wts, out, cfg1=cfg1,
                              cfg2=cfg2, align=align)


def unswizzle(sw, M, K):
    # the swizzle permutation (0, 1, 4, 3, 2, 5) is its own inverse
    return sw.reshape(1, M // 128, K // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(M, K)


def bench(fn, reps=15):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


if __name__ == "__main__":
    mode = sys.argv[1]
    out = {"mode": mode, "arch_override": os.environ.get("TRITON_OVERRIDE_ARCH")}
    if mode == "check":
        # (a) quant recipe, bit for bit against the CUDA kernel FlashInfer's MoE shares (cvt_warp_fp16_to_fp4)
        for g, scale in ((a1g_val, 1.0), (134.4, 3.0), (7.0, 0.01)):
            x = (torch.randn(600, H, device=dev) * scale).bfloat16()
            x[5, :32] = 0; x[7, 16:48] = -0.0
            qr, sr = ops.scaled_fp4_quant(x, torch.tensor([g], device=dev), is_sf_swizzled_layout=True)
            sr = unswizzle(sr.view(torch.uint8), 640, H // 16)[:600]
            qt = torch.empty(600, H // 2, dtype=torch.uint8, device=dev)
            st = torch.empty(600, H // 16, dtype=torch.float8_e4m3fn, device=dev)
            mf._quant_rows[((600 + 63) // 64, H // 64)](x, qt, st, 600, H=H, g=g, BM=64, BN=64)
            out[f"quant_g{g:g}"] = {"codes_mismatch": int((qt != qr).sum()), "sf_mismatch": int((st.view(torch.uint8) != sr).sum())}
        # (b) + (c) the whole MoE
        for M in (3456, 595, 128):
            x, ids, wts, xq, xs = inputs(M, M)
            with autotune(True):
                run_fi(xq, xs, ids, wts, M)
            yf = run_fi(xq, xs, ids, wts, M).float()
            yt = run_tr(xq, xs, ids, wts, M, poison=True)
            yt2 = run_tr(xq, xs, ids, wts, M, poison=True)
            torch.cuda.synchronize()
            out[str(M)] = {"rel_l2_vs_flashinfer": float((yt.float() - yf).norm() / yf.norm()),
                           "finite": bool(torch.isfinite(yt.float()).all()),
                           "bitstable": bool(torch.equal(yt, yt2)),
                           "fi_norm": float(yf.norm()), "tr_norm": float(yt.float().norm()),
                           "bit_identical_to_fi": bool(torch.equal(yt.float(), yf)),
                           "elems_differing": int((yt.float() != yf).sum())}
    else:
        Ms = [int(a) for a in sys.argv[2:]] or [3456]
        for M in Ms:
            x, ids, wts, xq, xs = inputs(M, M)
            with autotune(True):
                run_fi(xq, xs, ids, wts, M)
            al = moe_align_block_size(ids, mf.CFG1["BM"], E)
            r = {"fi_ms": bench(lambda: run_fi(xq, xs, ids, wts, M)),
                 "tr_ms": bench(lambda: run_tr(xq, xs, ids, wts, M)),
                 "tr_noalign_ms": bench(lambda: run_tr(xq, xs, ids, wts, M, align=al)),
                 "align_ms": bench(lambda: moe_align_block_size(ids, mf.CFG1["BM"], E))}
            out[str(M)] = r
    print(json.dumps(out))
