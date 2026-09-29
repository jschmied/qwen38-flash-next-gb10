#!/usr/bin/env python3
"""GDNNQ warm-replay check (agenda 2b): nvprobe's fixed 8k replay prompt, (a) max_tokens 1 — TTFT only, the prefill path
without any decoded text — cold once then warm three times, and (b) max_tokens 96 as nvprobe does (cold, warm).
Prints ONE json with the times and whether the 96-token replies are equal. argv: tag"""
import json, statistics, sys, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; tag = sys.argv[1]
unit = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")


def chat(content, n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                                "Authorization": "Bearer " + KEY})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return time.time() - t, out["choices"][0]["message"].get("content", "")


q1 = "[replay-1tok]\n" + unit * 220 + "\nList three risks in this module, one line each."
cold1, _ = chat(q1, 1)
warm1 = [chat(q1, 1)[0] for _ in range(3)]
q96 = "[replay-96tok]\n" + unit * 220 + "\nList three risks in this module, one line each."
cold96, a1 = chat(q96, 96)
warm96, a2 = chat(q96, 96)
print(json.dumps({"tag": tag, "tok1": {"cold_s": round(cold1, 3), "warm_s": [round(x, 3) for x in warm1],
                                       "warm_median_s": round(statistics.median(warm1), 3)},
                  "tok96": {"cold_s": round(cold96, 3), "warm_s": round(warm96, 3), "equal": a1 == a2}}))
