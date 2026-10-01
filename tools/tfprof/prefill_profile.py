#!/usr/bin/env python3
"""Profile one cold prompt prefill on TensorFold's Flash Next CUDA engine (one stream), for nsys with
--capture-range=cudaProfilerApi: a 1k warm-up prompt, then cudaProfilerStart, ONE N-token prompt with max_tokens=1,
cudaProfilerStop. Prompts: Python stdlib source with a unique first line (no cached prefix). Prints ONE json (wall).
argv: <tensorfold src> <model dir> <tokens>"""
import json, sys, sysconfig, time, uuid
from pathlib import Path

src, model, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:400]))


def prompt(k: int) -> list[int]:
    return tok.encode(f"# {uuid.uuid4().hex}\n") + ids[:k]


eng = FlashNextEngine(model, max_len=max(16384, n + 1024))
eng.generate(prompt(1024), 1, None, lambda t: None)                              # kernels built, graphs warm
torch.cuda.synchronize()
p = prompt(n)
torch.cuda.profiler.start()
t0 = time.perf_counter()
out = eng.generate(p, 1, None, lambda t: None)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
torch.cuda.profiler.stop()
print(json.dumps({"model": str(model), "tokens": len(p), "wall_s": round(wall, 3),
                  "stats": {k: v for k, v in (out or {}).items() if isinstance(v, (int, float, str))}}))
