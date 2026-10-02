#!/usr/bin/env python3
"""#260 review (path witness: prompt GEMM calls on the folded vs the W_q path): TensorFold Flash Next EXL3 prefill time (8k, 32k) and the first 16 tokens' hash, fixed prompts.
argv: <tensorfold src> <model dir> <label>"""
import hashlib, json, sys, sysconfig, time
from pathlib import Path

src, model, label = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402
from tensorfold.cuda.exl3 import prefill as x3p                                  # noqa: E402
CALLS = {"fold": 0, "wq": 0}
for _name, _key in (("tiles_fold", "fold"), ("tiles", "wq")):
    if hasattr(x3p, _name):
        def _wrap(*a, _f=getattr(x3p, _name), _k=_key):
            CALLS[_k] += 1
            return _f(*a)
        setattr(x3p, _name, _wrap)

tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))
eng = FlashNextEngine(model, max_len=40960)
eng.generate(tok.encode("# warm\n") + ids[:1024], 1, None, lambda t: None)
torch.cuda.synchronize()
for n in (8192, 32768):
    p = tok.encode(f"# prompt of {n} tokens\n") + ids[:n]          # the same prompt in every arm, new in each process
    got = []
    t0 = time.perf_counter()
    st = eng.generate(p, 16, None, lambda t: got.extend(t) and None, stop_eos=False)
    torch.cuda.synchronize()
    print(json.dumps({"label": label, "tokens": len(p), "prefill_s": round(st.get("prefill_s", -1), 3),
                      "wall_s": round(time.perf_counter() - t0, 3), "calls": dict(CALLS), "hash": hashlib.sha256(json.dumps(got).encode()).hexdigest()[:12]}),
          flush=True)
