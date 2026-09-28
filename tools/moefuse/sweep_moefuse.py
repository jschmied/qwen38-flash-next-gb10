#!/usr/bin/env python3
"""FNMOEFUSE tile sweep: times gemm1 (+SwiGLU+FP4) and gemm2 alone per config at one M, each config interleaved with
FlashInfer's whole MoE as the drift reference (ratios survive a shared GPU; absolute numbers need a quiet box).
argv: M stage(g1|g2) ; configs are the product below. Prints ONE json with per-config median ms and ms/fi."""
import itertools, json, os, sys
os.environ.setdefault("TRITON_OVERRIDE_ARCH", "sm120")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moefuse as T  # noqa: E402  (weights, inputs, run_fi)
import moe_fp4 as mf  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402

M, stage = int(sys.argv[1]), sys.argv[2]
x, ids, wts, xq, xs = T.inputs(M, M)
E, H, I, TOPK = T.E, T.H, T.I, T.TOPK


def timeit(fn, reps=7):
    for _ in range(2): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


res = {}
if stage == "g1":
    space = itertools.product((64, 128), (32, 64, 128), (128, 256), (4, 8), (2, 3, 4))
else:
    space = itertools.product((64, 128), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4))
for BM, BN, BK, nw, ns in space:
    if stage == "g1" and BM * BN * 2 > 128 * 128: continue
    if stage == "g2" and BM * BN > 128 * 128: continue
    sid, eid, ntpp = moe_align_block_size(ids, BM, E)
    NMB = eid.numel(); rows = sid.numel()
    iq = torch.zeros(rows, I // 2, dtype=torch.uint8, device="cuda")
    isf = torch.ones(rows, I // 16, dtype=torch.float8_e4m3fn, device="cuda")
    y = torch.empty(M * TOPK, H, dtype=torch.bfloat16, device="cuda")
    if stage == "g1":
        fn = lambda: mf._gemm1_swiglu_fp4[(NMB * (I // BN),)](
            xq, xs, T.w13, T.w13_s, sid, eid, ntpp, T.alpha1, T.a2g, iq, isf, M * TOPK, NMB,
            TOPK=TOPK, H=H, I=I, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    else:
        fn = lambda: mf._gemm2[(NMB * (H // BN),)](
            iq, isf, T.w2, T.w2_s, sid, eid, ntpp, T.alpha2, y, M * TOPK, NMB,
            H=H, I=I, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    key = f"BM{BM}_BN{BN}_BK{BK}_w{nw}_s{ns}"
    try:
        t = timeit(fn)
        f = timeit(lambda: T.run_fi(xq, xs, ids, wts, M))
        res[key] = {"ms": round(t, 3), "fi_ms": round(f, 3), "ratio": round(t / f, 4)}
    except Exception as ex:  # smem overflow and the like
        res[key] = {"err": type(ex).__name__ + ": " + str(ex)[:120]}
ok = {k: v for k, v in res.items() if "ratio" in v}
best = sorted(ok, key=lambda k: ok[k]["ratio"])[:8]
print(json.dumps({"M": M, "stage": stage, "best": [(k, ok[k]) for k in best], "n_ok": len(ok),
                  "n_err": len(res) - len(ok), "all": res}))
