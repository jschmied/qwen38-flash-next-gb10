#!/usr/bin/env python3
"""Quantize lm_head to blockwise FP8 (128x128), matching lovedheart's convention.

Convention verified empirically against the BF16 original:
    w_bf16 ~= w_fp8.float() * weight_scale_inv     (2.25% mean relative error)
i.e. `_inv` is a naming legacy -- the tensor holds the SCALE, not its reciprocal.
Storing the reciprocal here would produce fluent garbage, not a crash.

Builds a new checkpoint dir: everything hardlinked from the source (zero disk),
except the rewritten shard, the index, and hf_quant_config.json.
"""
import json, os, struct, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/opt/llm/models/qwen38-flash-next-fp8mix"
DST = "/opt/llm/models/qwen38-flash-next-fp8head"
TARGET = "lm_head.weight"
BN = BK = 128
FP8_MAX = 448.0

def blockwise_fp8(w):
    """w: [out, in] float32 -> (fp8 tensor, scale [out/BN, in/BK] float32)."""
    o, i = w.shape
    assert o % BN == 0 and i % BK == 0, f"{o}x{i} not divisible by {BN}x{BK}"
    wb = w.view(o // BN, BN, i // BK, BK)              # block view
    absmax = wb.abs().amax(dim=(1, 3))                 # [ob, ib]
    scale = (absmax / FP8_MAX).clamp(min=1e-12)
    q = (wb / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.view(o, i).to(torch.float8_e4m3fn), scale.to(torch.float32)

def main():
    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    shard = wm[TARGET]
    print(f"  target {TARGET} lives in {shard}")

    # hardlink everything except what we rewrite
    # config.json MUST be copied, not hardlinked: vLLM prefers its embedded
    # quantization_config over hf_quant_config.json, and a hardlink means an
    # edit here silently rewrites the source checkpoint through the same inode.
    rewrite = {shard, "model.safetensors.index.json", "hf_quant_config.json", "config.json"}
    n = 0
    for f in os.listdir(SRC):
        if f in rewrite or f.endswith(".aria2") or f == "slice.log": continue
        s, d = os.path.join(SRC, f), os.path.join(DST, f)
        if os.path.exists(d): os.unlink(d)
        try: os.link(s, d); n += 1
        except OSError: pass
    print(f"  hardlinked {n} files")

    # rewrite the shard
    tensors = {}
    with safe_open(os.path.join(SRC, shard), framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    w = tensors[TARGET].float()
    print(f"  quantizing {TARGET} {tuple(w.shape)} ...")
    q, scale = blockwise_fp8(w)
    rec = q.float() * scale.repeat_interleave(BN, 0).repeat_interleave(BK, 1)
    rel = ((rec - w).abs().mean() / w.abs().mean()).item()
    print(f"  round-trip mean relative error: {rel:.4%}   (lovedheart's layers: ~2.25%)")
    tensors[TARGET] = q
    tensors[TARGET.replace(".weight", ".weight_scale_inv")] = scale
    save_file(tensors, os.path.join(DST, shard), metadata=meta or {"format": "pt"})
    sz_old = os.path.getsize(os.path.join(SRC, shard)) / 2**30
    sz_new = os.path.getsize(os.path.join(DST, shard)) / 2**30
    print(f"  shard {sz_old:.2f} -> {sz_new:.2f} GiB")

    # index: add the scale entry
    wm[TARGET.replace(".weight", ".weight_scale_inv")] = shard
    idx["metadata"] = idx.get("metadata", {})
    json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"))
    print(f"  index updated ({len(wm)} entries)")

    # quant config: un-exclude lm_head, declare it FP8_PB_WO
    q = json.load(open(f"{SRC}/hf_quant_config.json"))
    qz = q["quantization"]
    ex = [e for e in qz.get("exclude_modules", []) if e != "lm_head"]
    qz["exclude_modules"] = ex
    qz["quantized_layers"]["lm_head"] = {"quant_algo": "FP8_PB_WO"}
    json.dump(q, open(f"{DST}/hf_quant_config.json", "w"), indent=2)
    print(f"  quant config: lm_head un-excluded, declared FP8_PB_WO "
          f"({len(qz['quantized_layers'])} layers)")

if __name__ == "__main__":
    main()
