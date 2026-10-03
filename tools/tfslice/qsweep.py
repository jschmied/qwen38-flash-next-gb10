#!/usr/bin/env python3
"""Block-FP8 prompt matmul at Flash Next shapes: ms, TFLOPS and output bits against a saved reference (another tile
configuration). argv: <tensorfold src> <save|check> <ref.pt>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear                         # noqa: E402

mode, refp = sys.argv[2], sys.argv[3]
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(5)
SHAPES = {"gdn_in_qkv": (10240, 2560), "gdn_in_z": (6144, 2560), "gdn_out": (2560, 6144), "attn_q": (12288, 2560),
          "attn_kv": (512, 2560)}


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


ref = torch.load(refp) if mode == "check" else {}
save, ok, tot = {}, True, 0.0
for name, (n, k) in SHAPES.items():
    w = (torch.randn(n, k, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    si = torch.rand(-(-n // 128), k // 128, device=dev, generator=g) * 0.01 + 0.001
    lin = Fp8BlockLinear.from_checkpoint(w, si)
    for m in (2048, 8192, 300):
        x = (torch.randn(m, k, device=dev, generator=g)).to(torch.bfloat16)
        y = lin.prefill(x)
        key = f"{name}-{m}"
        if mode == "save":
            save[key] = y.cpu()
            eq = None
        else:
            eq = bool(torch.equal(y.cpu().view(torch.int16), ref[key].view(torch.int16)))
            ok &= eq
        t = timed(lambda: lin.prefill(x))
        if m == 8192:
            tot += t
        print(json.dumps({"case": key, "ms": round(t, 3), "TF": round(2 * m * n * k / t / 1e9, 1), "eq": eq}), flush=True)
if mode == "save":
    torch.save(save, refp)
print(f"SUM8192 {tot:.2f} ms", "ALL_EQUAL" if ok else "MISMATCH")
