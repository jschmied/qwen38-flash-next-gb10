#!/usr/bin/env python3
"""ncu target: one 2,048-token prompt (one prompt chunk) on Flash Next, after a warm-up prompt. argv: <src> <model>"""
import sys, sysconfig
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402
from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

precision.set_mode("checkpoint", asked=True)
model = Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:200]))
eng = FlashNextEngine(model, max_len=8192)
eng.generate(ids[:2048], 1, None, lambda t: None)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
eng.generate(ids[2048:4096], 1, None, lambda t: None)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
