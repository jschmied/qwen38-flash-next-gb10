#!/usr/bin/env python3
"""Three fixed greedy prompts against the server on :8092 (64 tokens each, thinking off), then /health's drafted and
accepted totals. Prints ONE json. argv: <tag>"""
import json, sys, urllib.request
P = ["List the first ten prime numbers, separated by commas.", "What is the capital of France? Answer in one sentence.",
     "Write a Python function that returns the factorial of n."]
out = {"tag": sys.argv[1], "replies": []}
for p in P:
    b = {"model": "flashnext", "messages": [{"role": "user", "content": p}], "max_tokens": 64, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions",
            json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600).read())
        out["replies"].append((d["choices"][0]["message"]["content"] or "")[:200])
    except Exception as e:
        out["replies"].append(f"ERROR {type(e).__name__}: {e}"[:200])
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=30).read())
    out["health"] = {k: v for k, v in h.items() if any(s in k for s in ("drafted", "accepted", "token_sha", "rounds"))}
except Exception as e:
    out["health"] = str(e)
print(json.dumps(out))
