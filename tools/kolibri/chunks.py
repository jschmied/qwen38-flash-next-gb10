"""Kolibri on TensorFold: cold prefill tok/s of one prompt at several chunk sizes (the ring sized for the largest)."""

import sys
import time
from pathlib import Path

import torch

from tensorfold.families.kolibri1.cuda import forward
from tensorfold.families.kolibri1.cuda.weights import load

d = Path(sys.argv[1])
sizes = [int(x) for x in sys.argv[2].split(",")]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
forward.PROMPT_CHUNK = max(sizes)
w = load(d)
ids = torch.randint(1000, 120000, (n,), generator=torch.Generator().manual_seed(1)).tolist()
for rep in range(2):
    for c in sizes:
        m = forward.Model(w, n + 64, 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.prefill(ids, chunk=c)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"rep {rep} chunk {c}: {n} tokens in {dt:.2f}s = {n / dt:.0f} tok/s", flush=True)
        del m
        torch.cuda.empty_cache()
