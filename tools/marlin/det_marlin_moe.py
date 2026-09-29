#!/usr/bin/env python3
"""Is vLLM's Marlin NVFP4 MoE bit-deterministic? (2026-09-29; the `marlin` A/B changed greedy text between two starts.)

Flash-Next expert shapes (E=512, H=2560, I=640, top-k 10, group 16), random NVFP4 weights built with vLLM's own
`rand_marlin_weight_nvfp4_like`, fixed seeded inputs and routing. Per M in (1, 4, 55, 512):
  - repeat:  the same call 5x in this process, output hashes;
  - permute: the tokens reordered (outputs un-permuted), which changes the per-expert token order moe_align builds;
  - align:   whether moe_align_block_size's sorted_token_ids differ between calls.
Run the script twice (two processes) and compare the printed `xproc` hashes for the across-process case.
Prints ONE json object."""
import hashlib, json, sys
import torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_nvfp4_like
from vllm.scalar_type import scalar_types

E, H, I, TOPK, G = 512, 2560, 640, 10, 16
dev = "cuda"
torch.manual_seed(0)


def build(n, k):
    qs, ss, gs = [], [], []
    for _ in range(E):
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) / 20
        _, q, s, g = rand_marlin_weight_nvfp4_like(w, G)
        qs.append(q); ss.append(s); gs.append(g)
    return torch.stack(qs), torch.stack(ss), torch.stack(gs)


w1, s1, g1 = build(2 * I, H)
w2, s2, g2 = build(H, I)


def h(t):
    return hashlib.sha256(t.float().cpu().numpy().tobytes()).hexdigest()[:16]


def run(x, tw, ti):
    return fused_marlin_moe(hidden_states=x, w1=w1, w2=w2, bias1=None, bias2=None, w1_scale=s1, w2_scale=s2,
                            topk_weights=tw, topk_ids=ti, quant_type_id=scalar_types.float4_e2m1f.id,
                            global_num_experts=E, global_scale1=g1, global_scale2=g2)


out = {}
for M in (1, 4, 55, 512):
    gen = torch.Generator(device=dev).manual_seed(M)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=gen)
    ti = torch.stack([torch.randperm(E, device=dev, generator=gen)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev, generator=gen), -1).float()
    reps = [h(run(x, tw, ti)) for _ in range(5)]
    perm = torch.randperm(M, device=dev, generator=gen)
    y = run(x[perm], tw[perm], ti[perm])
    inv = torch.empty_like(perm); inv[perm] = torch.arange(M, device=dev)
    ref = run(x, tw, ti)
    yp = y[inv]
    ndiff = int((yp != ref).sum())
    aligns = set()
    for _ in range(5):
        st, _e, _n = moe_align_block_size(ti, 16, E, None, ignore_invalid_experts=True)
        aligns.add(hashlib.sha256(st.cpu().numpy().tobytes()).hexdigest()[:12])
    out[M] = {"repeat_classes": len(set(reps)), "xproc": reps[0], "permute_equal": ndiff == 0,
              "permute_diff_elems": ndiff, "permute_max_abs": float((yp.float() - ref.float()).abs().max()),
              "align_orders": len(aligns)}
print(json.dumps({"torch": torch.__version__, "results": out}))
sys.exit(0)
