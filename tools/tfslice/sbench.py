#!/usr/bin/env python3
"""Fused K slices for the bf16 / NVFP4 Triton matmuls: bit equality (fused == split, and split == a saved reference
from another tree) and time per call on the census shapes. argv: <tensorfold src> <save|check> <ref.pt>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4                      # noqa: E402

mode, refp = sys.argv[2], sys.argv[3]
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(7)
CASES = [("b16", 324, 10240, True), ("b16", 640, 2560, False), ("b16", 96, 2560, False), ("b16", 2560, 2560, False),
         ("b16", 2560, 6144, False), ("fp4", 2560, 640, False), ("fp4", 2560, 640, True)]
ROWS = (300, 2048, 2051)
W = {}
for kind, n, k, f32 in CASES:
    if (kind, n, k) in W:
        continue
    if kind == "b16":
        W[(kind, n, k)] = bf16.make_b16((torch.randn(n, k, generator=g, device=dev) * 0.05).to(torch.bfloat16))
    else:
        words = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
        scale = torch.randint(40, 60, (n, k // 16), dtype=torch.uint8, device=dev, generator=g).view(torch.float8_e4m3fn)
        W[(kind, n, k)] = nvfp4.make_fp4(words, scale, 0.01)


def run(kind, w, x, f32, fused):
    mod = bf16 if kind == "b16" else nvfp4
    mod.FUSED_ROWS = 1 if fused else 1 << 30
    return mod.matmul(x, w, f32=f32)


def timed(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


ref = torch.load(refp) if mode == "check" else {}
save = {}
ok_all = True
for kind, n, k, f32 in CASES:
    w = W[(kind, n, k)]
    for m in ROWS:
        x = (torch.randn(m, k, generator=g, device=dev)).to(torch.bfloat16)
        x[:, ::97] *= 8
        split = run(kind, w, x, f32, False)
        fused = run(kind, w, x, f32, True)
        key = f"{kind}-{n}-{k}-{int(f32)}-{m}"
        bits = lambda t: t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)  # noqa: E731
        same = torch.equal(bits(split), bits(fused))
        alone = all(torch.equal(bits(run(kind, w, x[r:r + 1].contiguous(), f32, False)[0]), bits(fused[r]))
                    for r in (0, m // 2, m - 1))
        if mode == "save":
            save[key] = split.cpu()
            vs_ref = None
        else:
            vs_ref = torch.equal(bits(split.cpu()), bits(ref[key]))
        t_split = timed(lambda: run(kind, w, x, f32, False))
        t_fused = timed(lambda: run(kind, w, x, f32, True))
        ok_all &= same and alone and vs_ref in (None, True)
        print(json.dumps({"case": key, "sk": (bf16.split_k(n, k) if kind == "b16" else nvfp4.split_for(n, k)),
                          "fused_eq_split": same, "row_alone_eq": alone, "split_eq_ref": vs_ref,
                          "split_ms": round(t_split, 3), "fused_ms": round(t_fused, 3)}), flush=True)
if mode == "save":
    torch.save(save, refp)
print("ALL_EQUAL" if ok_all else "MISMATCH")
