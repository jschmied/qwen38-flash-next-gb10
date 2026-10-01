#!/usr/bin/env python3
"""Is TensorFold's fp16 prompt matmul at BM 64 bit-identical to BM 16? Flash Next's F16 shapes, random weights.
argv: <tensorfold src>. Prints one JSON line a shape (equal, ms at both tiles)."""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm as M                      # noqa: E402

torch.manual_seed(0)
sc = M.Scratch(11)
sc.allocate("cuda", experts=None, rows=2048, ple_words=1, ple_heads=1, ple_dim=1) if False else None
# shapes: HC down/up mixes, GDN in_proj b|a, PLE key/value (n, k)
shapes = [(336, 10240), (10240, 320), (96, 2560), (2560, 1280), (256, 2560)]
for n, k in shapes:
    w = (torch.randn(n, k) * 0.02).half().cuda()
    sk = M.f16_split(n, k)
    part = torch.empty(sk * M.ROWS * n, dtype=torch.float32, device="cuda")
    f = M.F16(w, n, k, sk, type("S", (), {"part": part})())
    x = (torch.randn(2048, k) * 0.5).to(torch.bfloat16).cuda()
    a = torch.empty(2048, n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    f(x, a, 16)
    f(x, b, 64)
    t = {}
    for bm in (16, 64):
        o = torch.empty_like(a)
        for _ in range(3):
            f(x, o, bm)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20):
            f(x, o, bm)
        e1.record()
        torch.cuda.synchronize()
        t[bm] = e0.elapsed_time(e1) / 20
    print(json.dumps({"n": n, "k": k, "sk": sk, "equal": bool(torch.equal(a, b)), "ms_bm16": round(t[16], 3),
                      "ms_bm64": round(t[64], 3)}), flush=True)
