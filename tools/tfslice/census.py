#!/usr/bin/env python3
"""Which bf16 / NVFP4 Triton matmuls an 8K Flash Next prompt runs: (kind, N, K, rows, K slices, fp32 out) counts and
CUDA time per call shape (events around each call). argv: <tensorfold src> <model dir>"""
import collections, json, sys, sysconfig
from pathlib import Path

src, model = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4                      # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

ON = [False]
LOG = collections.defaultdict(lambda: [0, 0.0])
PENDING = []


def wrap(mod, kind, split):
    orig = mod.matmul

    def run(x, w, *a, **kw):
        if not ON[0]:
            return orig(x, w, *a, **kw)
        sk = kw.get("sk") or split(w.n, w.k)
        a0, a1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a0.record()
        r = orig(x, w, *a, **kw)
        a1.record()
        PENDING.append(((kind, w.n, w.k, x.shape[0], sk, bool(kw.get("f32", False))), a0, a1))
        return r
    mod.matmul = run


wrap(bf16, "b16", lambda n, k: bf16.split_k(n, k))
wrap(nvfp4, "fp4", nvfp4.split_for)
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))
eng = FlashNextEngine(model, max_len=16384)
eng.generate(tok.encode("# warm\n") + ids[:1024], 1, None, lambda t: None)
torch.cuda.synchronize()
ON[0] = True
st = eng.generate(tok.encode("# prompt of 8192 tokens\n") + ids[:8192], 1, None, lambda t: None, stop_eos=False)
torch.cuda.synchronize()
ON[0] = False
for key, a0, a1 in PENDING:
    LOG[key][0] += 1
    LOG[key][1] += a0.elapsed_time(a1)
print("PREFILL", st.get("prefill_s"))
for key, (n, ms) in sorted(LOG.items(), key=lambda kv: -kv[1][1]):
    print(json.dumps({"kind": key[0], "n": key[1], "k": key[2], "rows": key[3], "sk": key[4], "f32": key[5],
                      "calls": n, "ms": round(ms, 2)}))
