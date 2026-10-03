#!/usr/bin/env python3
"""NVFP4 grouped experts (gate|up SwiGLU, then down) at Flash Next's shapes and routing: time per layer and output
bits by prompt tile (16 / 32 / 64 pairs an item), and against a saved reference from another tree.
argv: <tensorfold src> <save|check> <ref.pt>"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda import experts as grouped                                  # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx                                # noqa: E402

mode, refp = sys.argv[2], sys.argv[3]
dev = "cuda"
E, TOP, D, NI = 512, 10, 2560, 640
g = torch.Generator(device=dev).manual_seed(11)


def proj(n, k):
    words = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    scales = torch.randint(0x28, 0x48, (E, n, k // 16), dtype=torch.uint8, device=dev, generator=g)
    return words, scales, (torch.rand(E, device=dev, generator=g) * 0.02 + 0.005)


ex = nvx.make(proj(NI, D), proj(NI, D), proj(D, NI))


def timed(fn, reps=5):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


ref = torch.load(refp) if mode == "check" else {}
save, ok = {}, True
tiles = [16] if mode == "save" else [16, 64]
for rows in (2048, 8192):
    x = (torch.randn(rows, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    picks = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOP] for _ in range(rows)]).to(torch.int32)
    for tile in tiles:
        plan = grouped.Plan(rows, TOP, E, dev, prefill=True)
        grouped.route(picks.contiguous(), plan, tile)
        act = torch.empty((rows * TOP, NI), dtype=torch.bfloat16, device=dev)
        y = torch.empty((rows * TOP, D), dtype=torch.bfloat16, device=dev)
        t_gu = timed(lambda: nvx.gate_up(x, ex, plan, act, rows))
        t_dn = timed(lambda: nvx.down(act, ex, plan, y, rows))
        key = f"{rows}"
        if mode == "save":
            save[key] = (act.cpu(), y.cpu())
            eq = None
        else:
            ra, ry = ref[key]
            eq = bool(torch.equal(act.cpu().view(torch.int16), ra.view(torch.int16)) and
                      torch.equal(y.cpu().view(torch.int16), ry.view(torch.int16)))
            ok &= eq
        flop = 2 * rows * TOP * (D * 2 * NI + NI * D)
        print(json.dumps({"rows": rows, "tile": tile, "gateup_ms": round(t_gu, 2), "down_ms": round(t_dn, 2),
                          "layer_ms": round(t_gu + t_dn, 2), "TFLOPS": round(flop / (t_gu + t_dn) / 1e9, 1),
                          "eq_ref": eq}), flush=True)
if mode == "save":
    torch.save(save, refp)
print("ALL_EQUAL" if ok else "MISMATCH")
