#!/usr/bin/env python3
"""Block-FP8 prompt matmul (Fp8BlockLinear, FP8G lane matmul) at Flash Next's dense shapes: ms and TFLOPS at 2,048 and
8,192 rows, against torch's bf16 matmul of the same shape. argv: <tensorfold src>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear                         # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(5)
SHAPES = {"gdn_in_qkv": (10240, 2560), "gdn_in_z": (6144, 2560), "gdn_out": (2560, 6144), "attn_q": (12288, 2560),
          "attn_o": (2560, 6144)}


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


for name, (n, k) in SHAPES.items():
    w = (torch.randn(n, k, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    si = torch.rand(-(-n // 128), k // 128, device=dev, generator=g) * 0.01 + 0.001
    lin = Fp8BlockLinear.from_checkpoint(w, si)
    wb = (torch.randn(k, n, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    for m in (2048, 8192):
        x = (torch.randn(m, k, device=dev, generator=g)).to(torch.bfloat16)
        t = timed(lambda: lin.prefill(x))
        tb = timed(lambda: x @ wb)
        f = 2 * m * n * k
        print(json.dumps({"shape": name, "n": n, "k": k, "rows": m, "fp8g_ms": round(t, 3), "fp8g_TF": round(f / t / 1e9, 1),
                          "bf16_ms": round(tb, 3), "bf16_TF": round(f / tb / 1e9, 1)}), flush=True)
