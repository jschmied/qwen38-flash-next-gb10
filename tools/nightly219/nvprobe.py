#!/usr/bin/env python3
"""Nightly validation probe: codeprobe (greedy/sampled code, prose, c4; hashes + accept) then TTFT 8k/30k with a
nonce (prefix cache cannot serve it) and a cache-hit replay (same 8k prompt twice: 2nd must be faster and equal).
Prints ONE json object. argv: <tag>"""
import json, statistics, subprocess, sys, time, urllib.request, uuid
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; tag = sys.argv[1]
p = subprocess.run([sys.executable, "/opt/llm/runners/ksweep/codeprobe.py", tag], capture_output=True, text=True)
out = json.loads([l for l in p.stdout.splitlines() if l.strip().startswith("{")][-1])


def chat(content, n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    return time.perf_counter() - t0, d["choices"][0]["message"]["content"]


unit = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")
tt = {}
for reps, name in ((220, "8k"), (860, "30k")):
    ts = [chat(f"[{uuid.uuid4()}]\n" + unit * reps + "\nSummarise what handler() does in one sentence.", 1)[0]
          for _ in range(3)]
    tt[name] = {"median_s": round(statistics.median(ts), 3), "all": [round(x, 3) for x in ts]}
q = "[replay-fixed]\n" + unit * 220 + "\nList three risks in this module, one line each."  # fixed: comparable across arms (§5z withdrawal)
c1, a1 = chat(q, 96); c2, a2 = chat(q, 96)
out["ttft"] = tt
out["replay"] = {"cold_s": round(c1, 3), "warm_s": round(c2, 3), "equal": a1 == a2}
print(json.dumps(out))
