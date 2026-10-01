#!/usr/bin/env python3
"""Flash Next NVFP4 on TensorFold: routed experts in the checkpoint's math vs bf16 rows (W4A16), one process per arm.
Prefill time of fixed 8k / 32k code prompts and their first 16 greedy tokens (hash), then decode: 256 greedy tokens
after a short code prompt, drafts on (tok/s and the reply's hash).
argv: <tensorfold src> <model dir> <label> <precision: checkpoint|full>"""
import hashlib, json, sys, sysconfig, time
from pathlib import Path

src, model, label, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

precision.set_mode(mode, asked=True)
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))
eng = FlashNextEngine(model, max_len=40960)
eng.generate(tok.encode("# warm\n") + ids[:1024], 1, None, lambda t: None)
torch.cuda.synchronize()


def h(xs):
    return hashlib.sha256(json.dumps(xs).encode()).hexdigest()[:12]


for n in (8192, 32768):
    p = tok.encode(f"# prompt of {n} tokens\n") + ids[:n]          # the same prompt in every arm, new in each process
    got = []
    t0 = time.perf_counter()
    st = eng.generate(p, 16, None, lambda t: got.extend(t) and None, stop_eos=False)
    torch.cuda.synchronize()
    print(json.dumps({"label": label, "mode": mode, "tokens": len(p), "prefill_s": round(st.get("prefill_s", -1), 3),
                      "wall_s": round(time.perf_counter() - t0, 3), "hash": h(got), "first": got[:16]}), flush=True)
chat = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that parses ISO 8601 durations "
                                 "like P3DT4H5M into seconds, with tests."}], tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
for rep in range(3):
    got = []
    t0 = time.perf_counter()
    st = eng.generate(tok.encode(chat, add_special_tokens=False), 256, None, lambda t: got.extend(t) and None,
                      stop_eos=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    dec = wall - st.get("prefill_s", 0.0)
    print(json.dumps({"label": label, "mode": mode, "decode_rep": rep, "tokens": len(got), "decode_s": round(dec, 3),
                      "tok_s": round(len(got) / dec, 2), "hash": h(got)}), flush=True)
