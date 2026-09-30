#!/usr/bin/env python3
"""FNHEAD4 unit test on the GPU with the real checkpoint head (qwen38-flash-next-mtpfp4 lm_head, FP8 128x128 blocks).
(1) exactness: the Triton W4A16 logits equal a float32 matmul against the NVFP4 dequant (the kernel computes what the
quantizer stored); (2) drift vs the FP8 head on hidden-state proxies: top-1 agreement, top-5 overlap, KL(FP8 || NVFP4)
at T=1. Proxies = random unit-RMS rows and the checkpoint's own embedding rows (unit-RMS normalized). Real hidden
states come from the server screen, not from here. ONE json; exit 1 if (1) fails."""
import json, os, sys, types
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_head4 as H
from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import dequantize_nvfp4_rows

M = "/opt/llm/models/qwen38-flash-next-mtpfp4"
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]


def load(name):
    with safe_open(f"{M}/{idx[name]}", "pt", device="cuda") as f:
        return f.get_tensor(name)


w = load("lm_head.weight"); s = load("lm_head.weight_scale_inv")
emb_name = next(k for k in idx if k.endswith("embed_tokens.weight"))
emb = load(emb_name)
V, K = w.shape
head = types.SimpleNamespace(weight=w, weight_scale=s, weight_block_size=[128, 128], tp_size=1)
lp = types.SimpleNamespace(org_vocab_size=V, soft_cap=None, scale=1.0)
model = types.SimpleNamespace(lm_head=head, logits_processor=lp)

st = H.build(model); model._fn_head4 = st
torch.manual_seed(0)
xr = torch.randn(32, K, device="cuda")
ids = torch.randint(0, emb.shape[0], (32,), device="cuda")
xe = emb[ids].float()
res = {}
for name, x in (("random", xr), ("embedding", xe)):
    x = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    got = H.target_logits(model, x).float()
    deq4 = dequantize_nvfp4_rows(st["q"], st["s"], st["g"])
    ref4 = x.float() @ deq4.T
    rel = float((got - ref4).abs().max() / ref4.abs().max())
    w8 = torch.cat([H._fp8_rows(head, r, min(r + 32768, V), 128, 128) for r in range(0, V, 32768)])
    ref8 = (x.float() @ w8.T)
    del w8, deq4
    p8 = torch.log_softmax(ref8, -1); p4 = torch.log_softmax(got, -1)
    kl = float((p8.exp() * (p8 - p4)).sum(-1).mean())
    t1 = float((ref8.argmax(-1) == got.argmax(-1)).float().mean())
    t5 = float(sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref8.topk(5).indices, got.topk(5).indices))
               / (5 * x.shape[0]))
    res[name] = {"kernel_vs_own_dequant_maxrel": rel, "top1_agree_vs_fp8": t1, "top5_overlap": t5, "kl_fp8_nvfp4": kl}
ok = all(r["kernel_vs_own_dequant_maxrel"] < 2e-2 for r in res.values())
print(json.dumps({"V": V, "K": K, "g": st["g"], "ok": ok, "res": res}))
sys.exit(0 if ok else 1)
