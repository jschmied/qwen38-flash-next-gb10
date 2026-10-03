#!/usr/bin/env python3
"""Ceiling for Flash Next's routed experts in a prompt piece on this GPU: bf16 batched matmuls with every expert's
rows padded to the piece's mean (512 experts, top 10, D 2560, I 640), gate|up then down, vs the measured TF kernel.
Rows per expert = piece rows x 10 / 512. Reports TFLOPS of useful work. argv: none"""
import json
import torch

E, TOP, D, I = 512, 10, 2560, 640
dev = "cuda"


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


wgu = torch.randn(E, D, 2 * I, device=dev, dtype=torch.bfloat16) * 0.02
wd = torch.randn(E, I, D, device=dev, dtype=torch.bfloat16) * 0.02
for piece in (2048, 4096, 8192):
    r = -(-piece * TOP // E)
    x = torch.randn(E, r, D, device=dev, dtype=torch.bfloat16)
    h = torch.randn(E, r, I, device=dev, dtype=torch.bfloat16)
    t_gu = timed(lambda: torch.bmm(x, wgu))
    t_d = timed(lambda: torch.bmm(h, wd))
    useful = 2 * piece * TOP * (D * 2 * I + I * D)                   # MACs x 2 of the real (unpadded) pairs
    print(json.dumps({"piece_rows": piece, "rows_per_expert": r, "gateup_ms": round(t_gu, 2), "down_ms": round(t_d, 2),
                      "per_layer_ms": round(t_gu + t_d, 2), "useful_TFLOPS": round(useful / (t_gu + t_d) / 1e9, 1),
                      "8k_prompt_48_layers_s": round((t_gu + t_d) * 48 * (8192 / piece) / 1000, 3)}), flush=True)
