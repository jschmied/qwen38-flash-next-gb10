#!/usr/bin/env python3
"""Decode with drafts over 8 prompts (4 code, 4 prose), 256 greedy tokens each: rounds, drafted / accepted, decode time,
per mode. argv: <tensorfold src> <model dir> <label> <mode>"""
import json, sys
from pathlib import Path

src, model, label, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

precision.set_mode(mode, asked=True)
TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "Write a Python class for a thread-safe bounded priority queue with blocking get and put, plus tests.",
         "Implement Dijkstra's shortest path in Rust with a binary heap and explain the complexity.",
         "Write a bash script that rotates log files older than seven days and compresses them.",
         "Explain in about 400 words why lighthouses were automated in the twentieth century.",
         "Describe the water cycle for a high-school student, with one concrete example per stage.",
         "Write a short story about a cartographer who discovers a town missing from every map.",
         "Compare the economic causes of the 1929 crash and the 2008 financial crisis."]
tok = AutoTokenizer.from_pretrained(model)
eng = FlashNextEngine(model, max_len=8192)
eng.generate(tok.encode("# warm\n" * 64), 32, None, lambda t: None, stop_eos=False)
tot = {"rounds": 0, "drafted": 0, "accepted": 0, "decode_s": 0.0, "tokens": 0}
for i, t in enumerate(TASKS):
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
    got = []
    st = eng.generate(ids, 256, None, lambda x: got.extend(x) and None, stop_eos=False)
    torch.cuda.synchronize()
    row = {k: st.get(k) for k in ("rounds", "drafted", "accepted", "decode_s")}
    row["tokens"] = len(got)
    for k in tot:
        tot[k] += row[k]
    print(json.dumps({"label": label, "mode": mode, "task": i, **row}), flush=True)
print("TOTAL " + json.dumps({"label": label, "mode": mode, **tot, "tok_s": round(tot["tokens"] / tot["decode_s"], 2),
                             "ms_per_round": round(1e3 * tot["decode_s"] / tot["rounds"], 2),
                             "accept": round(tot["accepted"] / tot["drafted"], 4),
                             "tokens_per_round": round(tot["tokens"] / tot["rounds"], 3)}), flush=True)
