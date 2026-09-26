#!/usr/bin/env python3
"""Concurrency probe (agenda round 2, item 3): N streams at once (N = 4, 8, 16), distinct prompts, greedy,
thinking off, 400 tokens. Each level runs twice; the second (warm) pass is kept. Records aggregate tok/s, per-stream
ms/token, acceptance and one hash per output, so arms can be checked for identical text before time is compared.
Prints ONE json object (armrun contract).  argv: <tag>"""
import hashlib, json, statistics, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8092/v1/chat/completions"; MET = "http://127.0.0.1:8092/metrics"; KEY = "sk-bench"
TOPICS = ["how a B-tree keeps itself balanced during inserts and deletes",
          "profiling a slow Python web service step by step",
          "how TCP congestion control reacts to packet loss",
          "optimistic versus pessimistic locking in databases"]
ANGLES = ["for a beginner", "for an expert reviewer", "with a worked example", "as a checklist"]
PROMPTS = [f"Explain {t} {a}, in about 300 words." for a in ANGLES for t in TOPICS]   # 16 distinct


def metrics():
    m = {}
    for ln in urllib.request.urlopen(MET, timeout=30).read().decode().splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        k, v = ln.rsplit(" ", 1); k = k.split("{")[0]
        try:
            m[k] = m.get(k, 0.0) + float(v)
        except ValueError:
            pass
    return m


def stream(prompt, out, i):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 400, "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); times = []; text = []
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            ln = raw.decode().strip()
            if not ln.startswith("data:") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[5:])
            if d.get("choices"):
                c = d["choices"][0].get("delta", {}).get("content")
                if c:
                    times.append(time.perf_counter()); text.append(c)
    s = "".join(text)
    out[i] = dict(start=t0, first=times[0] if times else None, last=times[-1] if times else None,
                  chunks=len(times), sha=hashlib.sha256(s.encode()).hexdigest()[:16])


def level(n):
    prompts = PROMPTS[:n]; m0 = metrics(); out = [None] * n
    th = [threading.Thread(target=stream, args=(p, out, i)) for i, p in enumerate(prompts)]
    for t in th: t.start()
    for t in th: t.join()
    m1 = metrics()
    d = lambda k: m1.get(k, 0.0) - m0.get(k, 0.0)
    gen = d("vllm:generation_tokens_total")
    D, A = d("vllm:spec_decode_num_drafts_total"), d("vllm:spec_decode_num_accepted_tokens_total")
    wall = max(o["last"] for o in out) - min(o["first"] for o in out)
    per = [1000 * (o["last"] - o["first"]) / max(1, o["chunks"] - 1) for o in out if o["chunks"] > 1]
    return dict(n=n, gen_tokens=gen, agg_tok_s=round(gen / wall, 2) if wall > 0 else None,
                stream_ms_per_chunk_median=round(statistics.median(per), 3),
                accept_len=round(1 + A / D, 3) if D else None, shas=[o["sha"] for o in out])


res = {}
for n in (4, 8, 16):
    level(n)                       # warm pass (same prompts): graphs, caches, JIT
    res[f"c{n}"] = level(n)
print(json.dumps(res))
