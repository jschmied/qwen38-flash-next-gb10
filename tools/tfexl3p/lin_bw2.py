#!/usr/bin/env python3
"""Achieved trellis bandwidth of TensorFold's EXL3 dense linear at decode rows (Flash Next's 5-bit shapes).
argv: <tensorfold src>. One JSON line per (shape, rows)."""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear                               # noqa: E402

torch.manual_seed(0)
dev = "cuda"
K2 = 10                                                                          # 5 bits
shapes = {"gdn.in_proj_qkv": (2560, 10240), "gdn.in_proj_z": (2560, 6144), "gdn.out_proj": (6144, 2560),
          "attn.q_proj": (2560, 12288), "attn.o_proj": (6144, 2560), "attn.k_proj": (2560, 512)}
for name, (K, N) in shapes.items():
    nbytes = K * N * K2 / 16
    pool = max(4, int(-(-120e6 // nbytes)))                  # distinct matrices: > 120 MB, far past the 24 MB L2
    lins = []
    for _ in range(pool):
        tr = torch.randint(-32768, 32767, (K // 16, N // 16, 8 * K2), dtype=torch.int16, device=dev)
        suh = (torch.randint(0, 2, (K,), device=dev).half() * 2 - 1)
        svh = (torch.randint(0, 2, (N,), device=dev).half() * 2 - 1)
        lins.append(Exl3Linear.from_tensors(tr, suh, svh, "mul1", None, dev))
    for R in (1, 4, 6, 17):
        x = (torch.randn(R, K, device=dev) * 0.5).to(torch.bfloat16)
        out = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
        for i in range(2 * pool):
            lins[i % pool](x, out)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for i in range(200):
            lins[i % pool](x, out)
        b.record()
        torch.cuda.synchronize()
        us = 1000 * a.elapsed_time(b) / 200
        print(json.dumps({"m": name, "K": K, "N": N, "R": R, "us": round(us, 1), "MB": round(nbytes / 1e6, 2),
                          "GBps": round(nbytes / us / 1e3, 1), "pool": pool, "hash": __import__("hashlib").sha256(lins[0](x, out).float().cpu().numpy().tobytes()).hexdigest()[:12]}), flush=True)
