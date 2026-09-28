#!/usr/bin/env python3
"""Byte-ledger capture (TODO 5b). Warm-up: one ~7.5k-token prefill and one short decode (not profiled, so compile and
autotune shapes are hot). Then /start_profile, one ~7.5k-token prefill (max_tokens 1, unique prompt), one decode window
(short prompt, 64 tokens), /stop_profile. The nsys wrapper uses --capture-range-end=stop-shutdown, so the server exits
right after /stop_profile and that call fails; that is expected. Prints ONE json. argv: <tag>"""
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8092"; KEY = "sk-bench"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
unit = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")


def chat(content, n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(b).encode(), H),
                                          timeout=1800).read())
    return round(time.perf_counter() - t0, 3), d["usage"]


def post(path):
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + path, b"{}", H), timeout=600).read(); return "ok"
    except Exception as e:  # /stop_profile: nsys stops the server (stop-shutdown)
        return type(e).__name__


out = {"warm_prefill": chat("[ledger-warm]\n" + unit * 220 + "\nSummarise handler() in one sentence.", 1),
       "warm_decode": chat("[ledger-warm-d] Write a Python function that parses ISO dates.", 32)}
out["start"] = post("/start_profile")
out["prefill"] = chat("[ledger-prefill]\n" + unit * 220 + "\nSummarise handler() in one sentence.", 1)
out["decode"] = chat("[ledger-decode] Write a Python function that merges two sorted lists.", 64)
out["stop"] = post("/stop_profile")
print(json.dumps(out))
