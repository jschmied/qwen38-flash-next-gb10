#!/usr/bin/env python3
"""Decode-side host cost on TensorFold's Flash Next CUDA engine. One process per arm.
argv: <tensorfold src> <model dir> <arm: S1|S0|M2|M4|L4> <tokens> [nsys]
Warm-up: every stream decodes 64 tokens. Then (nsys: cudaProfilerStart) all streams decode <tokens> greedy on fixed
code prompts at once, (cudaProfilerStop). Prints one json line starting with ARM."""
import hashlib, json, sys, threading, time
from pathlib import Path

src, model, arm, n = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
prof = len(sys.argv) > 5 and sys.argv[5] == "nsys"
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "Write a Python command-line tool that parses an nginx access log and prints the top 20 paths by bytes.",
         "Write a Python class for a thread-safe bounded priority queue with blocking get and put, plus tests.",
         "Write a Python implementation of Dijkstra's algorithm on a grid with obstacles, with a small CLI."]
streams = {"S1": 1, "S0": 1, "M2": 2, "M4": 4, "L4": 4}[arm]
active = 1 if arm == "L4" else streams      # L4: a streams=4 engine serving one request
tok = AutoTokenizer.from_pretrained(model)
prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False), add_special_tokens=False) for t in TASKS[:active]]
assert all(isinstance(p, list) and p and isinstance(p[0], int) for p in prompts)
eng = FlashNextEngine(model, max_len=16384, streams=streams, graphs=(arm != "S0"))


def run(count):
    outs, stats = [None] * active, [None] * active

    errs = []

    def one(i):
        got = []
        try:
            stats[i] = eng.generate(prompts[i], count, None, lambda t: got.extend(t) and None, stop_eos=False)
        except BaseException as x:                   # noqa: BLE001  re-raised on the main thread
            errs.append(x)
        outs[i] = got
    th = [threading.Thread(target=one, args=(i,)) for i in range(active)]
    t0 = time.perf_counter()
    [t.start() for t in th]
    [t.join() for t in th]
    torch.cuda.synchronize()
    if errs:
        raise errs[0]
    return time.perf_counter() - t0, outs, stats


run(64)
if prof:
    torch.cuda.profiler.start()
wall, outs, stats = run(n)
if prof:
    torch.cuda.profiler.stop()
total = sum(len(o) for o in outs)
keep = lambda s: {k: v for k, v in (s or {}).items() if isinstance(v, (int, float, str))}   # noqa: E731
print("ARM " + json.dumps({"arm": arm, "streams": streams, "graphs": arm != "S0", "nsys": prof, "wall_s": round(wall, 3),
                           "tokens": total, "tok_s": round(total / wall, 2), "active": active, "ms_tok_per_stream": round(1000 * wall * active / total, 3),
                           "hashes": [hashlib.sha256(json.dumps(o).encode()).hexdigest()[:12] for o in outs],
                           "lens": [len(o) for o in outs], "stats": [keep(s) for s in stats]}), flush=True)
