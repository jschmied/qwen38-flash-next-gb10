"""What verifying k draft rows costs on Kolibri-1 (TensorFold kolibri1, decode path), against one decode row.

    python verifycost.py MODEL_DIR TEXT_PREFIX [--contexts 1024,8192,32768] [--rows 1,2,3,4,6] [--reps 30]

TEXT_PREFIX: data_swe.py-format tokens; a real conversation gives the context and the verified rows, so the routed
experts are those real text picks. Each cell: the median over ``reps`` of one ``forward(prompt=False)`` with a chain of
k rows at the context's end (the call the decoder makes per verify round), CUDA-synchronised. Also: the drafter's
output-head read for one row, full vocabulary vs a 32k slice (bf16), the per-draft-step cost a learned drafter adds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "drafter"))
from train_mt import conversations  # noqa: E402


def timed(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


@torch.no_grad()
def main() -> None:
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model
    from tensorfold.families.kolibri1.cuda.weights import load

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("text")
    ap.add_argument("--contexts", default="1024,8192,32768")
    ap.add_argument("--rows", default="1,2,3,4,6")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    ctxs = [int(v) for v in a.contexts.split(",")]
    ks = [int(v) for v in a.rows.split(",")]
    toks = np.concatenate([t for _, t in conversations(a.text)])[: max(ctxs) + max(ks) + 8].astype(np.int64).tolist()
    w = load(a.model_dir)
    model = Model(w, max(ctxs) + max(ks) + 64, 1)
    res = {}
    for c in ctxs:
        model.prefill(toks[:c])
        base = None
        for k in ks:
            ms = timed(lambda: model.forward([Chain(0, c, toks[c:c + k])], prompt=False, rows=range(k)), a.reps)
            base = base or ms
            res.setdefault(str(c), {})[str(k)] = {"ms": round(ms, 2), "x_one_row": round(ms / base, 3)}
            print(json.dumps({"context": c, "rows": k, "ms": round(ms, 2), "x_one_row": round(ms / base, 3)}), flush=True)
    x = torch.randn(1, w.head.shape[1], device=w.head.device, dtype=torch.bfloat16)
    head = w.head.to(torch.bfloat16)
    full = timed(lambda: x @ head.T, a.reps)
    part = timed(lambda: x @ head[:32768].T, a.reps)
    res["draft_head_ms"] = {"full_vocab": round(full, 3), "slice_32k": round(part, 3)}
    print(json.dumps(res["draft_head_ms"]), flush=True)
    print(json.dumps({"result": res}), flush=True)


if __name__ == "__main__":
    main()
