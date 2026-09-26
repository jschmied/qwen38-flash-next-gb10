#!/usr/bin/env python3
"""Item 7: sustained c=4 decode for ~4 min (4 streams x 3500 tokens, thinking off, greedy). Every streamed chunk is
timestamped; throughput per 5 s window (all streams), dips = windows < 70 % of the median. Compaction counters from
/proc/vmstat before/after. Also the c=1/c=4 decprobe numbers. ONE json. argv: <tag>"""
import json, statistics, subprocess, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; tag = sys.argv[1]
P = ["Write a long, detailed technical history of the B-tree and its variants, with examples, as long as you can.",
     "Write a long, detailed guide to profiling and optimising Python web services, as long as you can.",
     "Write a long, detailed explanation of TCP congestion control algorithms and their evolution, as long as you can.",
     "Write a long, detailed comparison of database concurrency control schemes, as long as you can."]
def vm():
    d = {}
    for ln in open("/proc/vmstat"):
        k, v = ln.split()
        if k.startswith("compact_") or k in ("pgmigrate_success", "thp_migration_success"):
            d[k] = int(v)
    return d
times = []; lock = threading.Lock()
def stream(p):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 3500, "stream": True, "ignore_eos": True,
                    "messages": [{"role": "user", "content": p}], "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            ln = raw.decode().strip()
            if ln.startswith("data:") and ln != "data: [DONE]":
                d = json.loads(ln[5:])
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    with lock: times.append(time.perf_counter())
dec = json.loads(subprocess.run([sys.executable, "/opt/llm/runners/fullcg/decprobe.py", tag], capture_output=True, text=True).stdout.strip().splitlines()[-1])
v0 = vm(); th = [threading.Thread(target=stream, args=(p,)) for p in P]
for t in th: t.start()
for t in th: t.join()
v1 = vm()
times.sort(); t0 = times[0] + 10; t1 = times[-1] - 5                  # skip ramp-up and tail
w = []; x = t0
while x + 5 <= t1:
    w.append(sum(1 for t in times if x <= t < x + 5) / 5.0); x += 5
med = statistics.median(w)
res = {"c1_ms_tok": dec["c1"]["decode_ms_per_tok"], "c4_tok_s": dec["c4"]["agg_tok_s"], "c1_shas": dec["c1"]["shas"],
       "sustain": {"windows": len(w), "chunks_per_s_median": round(med, 2), "p10": round(sorted(w)[len(w) // 10], 2),
                   "min": round(min(w), 2), "dips_below_70pct": sum(1 for y in w if y < 0.7 * med),
                   "mean": round(statistics.mean(w), 2), "span_s": round(t1 - t0, 1)},
       "vmstat_delta": {k: v1[k] - v0.get(k, 0) for k in v1}}
print(json.dumps(res))
