#!/usr/bin/env python3
"""#260 on Flash Next shapes: EXL3 dense linear time by row count, linear_kernel (MODE 0) vs mid-M (6) vs linear_wc (7,
4/6-bit mul1 only), effective weight GB/s against a device copy; and the bf16 GEMM peak the folded prompt GEMM is held
to. Random trellis words (timing only), 4 weight copies rotated (L2 bust), CUDA events over 40 calls.
argv: <tensorfold src>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import linear as L                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"


def ev_time(fn, reps=40):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(reps):
        fn(i)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


big = torch.empty(1 << 28, dtype=torch.uint8, device=dev); dst = torch.empty_like(big)
copy_ms = ev_time(lambda i=0: dst.copy_(big), 20)
copy_gbs = 2 * big.numel() / copy_ms / 1e6                                        # read + write
print(json.dumps({"copy_GBs_rw": round(copy_gbs, 1)}), flush=True)

SHAPES = {"gdn_in_qkv": (2560, 10240), "gdn_in_z": (2560, 6144), "gdn_out": (6144, 2560), "attn_q": (2560, 12288),
          "attn_o": (6144, 2560), "lm_head": (2560, 248320)}
for bits in (5, 4):
    for name, (k, n) in SHAPES.items():
        if bits == 4 and name not in ("gdn_in_qkv", "gdn_out"):
            continue
        copies = 2 if name == "lm_head" else 4
        lins = [L.Exl3Linear.from_tensors(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16),
                                          (torch.randint(0, 2, (k,)) * 2 - 1).half(), (torch.rand(n) * 0.01).half(), "mul1", None, dev)
                for _ in range(copies)]
        wbytes = k * n * bits / 8
        for rows in (1, 7, 16, 17, 28, 32, 56, 64, 128):
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
            res = {"bits": bits, "shape": name, "k": k, "n": n, "rows": rows}
            ref = None
            for mode in (0, 6, 7):
                if mode == 7 and bits not in (4, 6):
                    continue
                L.MODE = mode
                ms = ev_time(lambda i=0: lins[i % copies](x, out=out))
                lins[0](x, out=out); torch.cuda.synchronize()
                ref = out.clone() if ref is None else ref
                res[f"m{mode}_us"] = round(ms * 1000, 1)
                res[f"m{mode}_GBs"] = round(wbytes / ms / 1e6, 1)
                res[f"m{mode}_same"] = bool(torch.equal(out, ref))
            print(json.dumps(res), flush=True)

for m, k, n in ((8192, 2560, 10240), (8192, 6144, 2560), (2048, 2560, 10240)):
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16); w = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    ms = ev_time(lambda i=0: a @ w, 20)
    print(json.dumps({"bf16_mm": [m, k, n], "TFLOPs": round(2 * m * k * n / ms / 1e9, 1)}), flush=True)
