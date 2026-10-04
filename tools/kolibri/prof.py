"""Kolibri on TensorFold: per-kernel GPU time of a decode step and of a prompt chunk (torch profiler)."""

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tensorfold.families.kolibri1.cuda.forward import Chain, Model
from tensorfold.families.kolibri1.cuda.weights import load

d = Path(sys.argv[1])
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
w = load(d)
m = Model(w, 16384, 1)
ids = torch.randint(1000, 120000, (4096,), generator=torch.Generator().manual_seed(0)).tolist()
m.prefill(ids[:2000])
pos = 2000
for _ in range(5):                                  # warm
    m.step([5], [pos], [0]); pos += 1
torch.cuda.synchronize()


def table(p, label, n):
    agg = defaultdict(float)
    for e in p.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
            agg[e.key] += e.self_device_time_total
    tot = sum(agg.values())
    print(f"== {label}: GPU {tot / n / 1000:.2f} ms per call")
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:18]:
        print(f"  {v / n / 1000:7.3f} ms  {100 * v / tot:5.1f}%  {k[:90]}")


t0 = time.perf_counter()
for _ in range(20):
    m.step([5], [pos], [0]); pos += 1
torch.cuda.synchronize()
print(f"decode wall {1000 * (time.perf_counter() - t0) / 20:.2f} ms per token")
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        m.step([5], [pos], [0]); pos += 1
    torch.cuda.synchronize()
table(p, "decode step", 10)
m2 = Model(w, 16384, 1)
torch.cuda.synchronize()
t0 = time.perf_counter()
m2.forward([Chain(0, 0, ids[:rows])], prompt=True)
torch.cuda.synchronize()
print(f"prompt chunk {rows}: wall {time.perf_counter() - t0:.3f} s")
m3 = Model(w, 16384, 1)
with profile(activities=[ProfilerActivity.CUDA]) as p:
    m3.forward([Chain(0, 0, ids[:rows])], prompt=True)
    torch.cuda.synchronize()
table(p, f"prompt chunk {rows}", 1)
