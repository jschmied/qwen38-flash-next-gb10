#!/usr/bin/env python3
"""What does the UE8M0 requant cost OUR checkpoint's 157 FP8_PB_WO layers?

Baseline is what CUTLASS actually consumes: the original FP8 weights with their fp32 block
scales. The DeepGEMM path consumes the same weights after requant_weight_ue8m0_inplace. The
divergence between the two dequantized tensors IS the added error -- no GSM8K run needed to
size it.

CPU-only by default: a GPU tensor job contends for unified memory bandwidth and would skew a
live A/B (gb10-quant-speed-q8-vs-q5). Run only when no fx-* / dgemmdrv unit is active.
"""
import argparse, json, os, sys, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/opt/llm/models/qwen38-flash-next-mtpfp4")
    ap.add_argument("--layers", type=int, default=12, help="how many FP8_PB_WO tensors to sample")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        requant_weight_ue8m0_inplace)

    hq = json.load(open(os.path.join(a.model, "hf_quant_config.json")))
    q = hq.get("quantization", hq)
    ql = q.get("quantized_layers") or {}
    pbwo = [k for k, v in ql.items()
            if (v.get("quant_algo") if isinstance(v, dict) else v) == "FP8_PB_WO"]
    print(f"FP8_PB_WO layers in checkpoint: {len(pbwo)}; sampling {a.layers}")

    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    rows, done = [], 0
    for name in pbwo:
        wk = f"{name}.weight"
        sk = next((f"{name}.{suf}" for suf in ("weight_scale_inv", "weight_scale")
                   if f"{name}.{suf}" in idx), None)
        if wk not in idx or sk is None:
            continue
        with safe_open(os.path.join(a.model, idx[wk]), framework="pt") as f:
            w = f.get_tensor(wk)
        with safe_open(os.path.join(a.model, idx[sk]), framework="pt") as f:
            s = f.get_tensor(sk)
        if w.dtype != torch.float8_e4m3fn or w.dim() != 2:
            continue
        w, s = w.to(a.device), s.to(a.device).float()

        def deq(wq, sc):
            M, K = wq.shape
            bm, bk = 128, 128
            e = torch.repeat_interleave(torch.repeat_interleave(sc, bm, 0), bk, 1)[:M, :K]
            return wq.to(torch.float32) * e

        ref = deq(w, s)                       # what CUTLASS consumes
        w2, s2 = w.clone(), s.clone()
        requant_weight_ue8m0_inplace(w2, s2, block_size=(128, 128))
        got = deq(w2, s2)                     # what DeepGEMM consumes

        d = got - ref
        rel = (d.norm() / ref.norm().clamp(min=1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().double(), got.flatten().double(), dim=0).item()
        rows.append((name, tuple(w.shape), rel, 1 - cos))
        print(f"  {name[:58]:58} {str(tuple(w.shape)):16} rel_fro={rel:.5f}  1-cos={1-cos:.3e}")
        done += 1
        if done >= a.layers:
            break

    if rows:
        r = torch.tensor([x[2] for x in rows])
        c = torch.tensor([x[3] for x in rows])
        print(f"\nsampled {len(rows)} layers")
        print(f"  rel_fro   mean={r.mean():.5f}  min={r.min():.5f}  max={r.max():.5f}")
        print(f"  1-cos     mean={c.mean():.3e}  max={c.max():.3e}")
        print("\nreference: NVFP4 weights sit 4.5x further from BF16 than FP8 "
              "(memory nvfp4-quantization-cost-measured) -- compare against that scale.")

main()
