"""Deterministic split-K (fn_bf16sk) vs cuBLAS, M in {1, 4, 16}; weights rotated >= 96 MiB; CUDA graph of 48 calls.
Also: bitwise repeatability over 20 calls, and max rel error vs F.linear."""
import json, math, sys, itertools, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk import bf16sk
from bf16mb import bench

SHAPES = {"mixer_down": (324, 10240), "mixer_up": (10240, 320), "in_proj_ba": (96, 2560), "router": (512, 2560),
          "shared_gate_up": (1280, 2560), "shared_down": (2560, 640)}
res = []
for name, (N, K) in SHAPES.items():
    nbytes = N * K * 2; floor = nbytes / 220e9 * 1e6
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(max(48, math.ceil(96 * 2**20 / nbytes)))]
    for M in (1, 4, 16):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x, ws[0]).float()
        row = {"shape": name, "M": M, "floor_us": round(floor, 1),
               "cublas_us": bench(lambda x, w, e: F.linear(x, w), x, ws, None), "cands": []}
        for bn, bk, s in itertools.product((16, 32, 64), (64, 128, 256), (1, 2, 4, 8, 16)):
            if K // s < bk:
                continue
            f = lambda x, w, e, c=(bn, bk, s): bf16sk(x, w, *c)
            try:
                got = f(x, ws[0], None).float()
                err = ((got - ref).abs().max() / ref.abs().max()).item()
                outs = {f(x, ws[0], None).view(torch.int16).sum().item() for _ in range(20)}
                row["cands"].append({"cfg": [bn, bk, s], "us": bench(f, x, ws, None), "err": round(err, 5),
                                     "repeatable": len(outs) == 1})
            except Exception as ex:
                row["cands"].append({"cfg": [bn, bk, s], "error": str(ex)[:80]})
        ok = [c for c in row["cands"] if "us" in c and c["repeatable"] and c["err"] < 1e-2]
        row["best"] = min(ok, key=lambda c: c["us"]) if ok else None
        b = row["best"]
        print(f"{name:15s} M={M:2d} floor {floor:6.1f} cublas {row['cublas_us']:6.1f} best {b['cfg'] if b else None} "
              f"{b['us'] if b else '-'} ({round(b['us']/floor,2) if b else '-'}x) err {b['err'] if b else '-'}", flush=True)
        res.append(row)
    del ws; torch.cuda.empty_cache()
json.dump(res, open(sys.argv[1], "w"), indent=1)
