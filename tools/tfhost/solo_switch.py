#!/usr/bin/env python3
"""Does the lone-stream graph slot keep its graphs when consecutive lone requests have different prompts?
A streams=4 engine; requests A, B, C, A' one after another (A' extends A's prompt), 256 tokens each, greedy; graph
captures and ms/token per request. argv: <tensorfold src> <model dir>"""
import hashlib, json, sys, time
from pathlib import Path

src, model = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda import graphs as G                       # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

CAPS = []
_orig = G.Graphs._capture


def _timed(self, fn):
    t = time.perf_counter()
    try:
        return _orig(self, fn)
    finally:
        CAPS.append(1000 * (time.perf_counter() - t))


G.Graphs._capture = _timed
tok = AutoTokenizer.from_pretrained(model)
TASKS = {"A": "Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "B": "Write a Python command-line tool that parses an nginx access log and prints the top 20 paths by bytes.",
         "C": "Write a Python class for a thread-safe bounded priority queue with blocking get and put, plus tests."}
enc = lambda t: tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,  # noqa: E731
                                                   add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
eng = FlashNextEngine(model, max_len=16384, streams=4)
print(json.dumps({"captures_at_start": len(CAPS)}), flush=True)
first_a = None
for name in ("A", "B", "C", "A2"):
    if name == "A2":
        prompt = first_a + enc("Now add a method that reports the hit rate.")[0:0] + tok.encode("\n\nAlso report hit rate.")
    else:
        prompt = enc(TASKS[name])
    before = len(CAPS)
    got = []
    t0 = time.perf_counter()
    st = eng.generate(prompt, 256, None, lambda t: got.extend(t) and None, stop_eos=False)
    wall = time.perf_counter() - t0
    if name == "A":
        first_a = prompt + got
    print(json.dumps({"request": name, "ms_tok": round(1000 * wall / max(1, len(got)), 3), "tokens": len(got),
                      "captures": len(CAPS) - before, "capture_ms": round(sum(CAPS[before:]), 1),
                      "cached": st.get("cached"), "hash": hashlib.sha256(json.dumps(got).encode()).hexdigest()[:12]}),
          flush=True)
