#!/usr/bin/env python3
"""Drafted == serial on the real checkpoint: 4 prompts x 160 greedy tokens with MTP drafts and without, one mode.
argv: <tensorfold src> <model dir> <mode>"""
import json, sys
from pathlib import Path

src, model, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, src)
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

precision.set_mode(mode, asked=True)
TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "Explain in about 400 words why lighthouses were automated in the twentieth century.",
         "Implement Dijkstra's shortest path in Rust with a binary heap and explain the complexity.",
         "Write a short story about a cartographer who discovers a town missing from every map."]
tok = AutoTokenizer.from_pretrained(model)
eng = FlashNextEngine(model, max_len=8192)
same = 0
for i, t in enumerate(TASKS):
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
    out = {}
    for draft in (True, False):
        got = []
        eng.generate(ids, 160, None, lambda x: got.extend(x) and None, draft=draft, stop_eos=False)
        out[draft] = got
    eq = out[True] == out[False]
    same += eq
    first = next((j for j, (a, b) in enumerate(zip(out[True], out[False])) if a != b), None)
    print(json.dumps({"mode": mode, "task": i, "equal": eq, "first_diff": first, "n": len(out[True])}), flush=True)
print(f"RESULT {mode}: drafted == serial {same}/{len(TASKS)}", flush=True)
