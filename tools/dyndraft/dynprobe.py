#!/usr/bin/env python3
"""FNDYN threshold sweep inside ONE server start (thr 0 = off is the interleaved control). Derived from the #58449 probe: decode speed with MTP n=3, measured on streamed output (first -> last token), at c=1
and c=4, then the 8-turn agent loop. Records a hash of every output and the acceptance counters, so arms can be
checked for identical text before any time is compared. Prints ONE json object (armrun contract).  argv: <tag>"""
import hashlib, json, re, statistics, subprocess, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; MET = "http://127.0.0.1:8092/metrics"; KEY = "sk-bench"
PROMPTS = [
    "Explain in about 500 words how a B-tree keeps itself balanced during inserts and deletes.",
    "Write a detailed, step-by-step guide (about 500 words) to profiling a slow Python web service.",
    "Describe in about 500 words how TCP congestion control reacts to packet loss, with examples.",
    "Write about 500 words comparing optimistic and pessimistic locking in databases, with trade-offs.",
]
tag = sys.argv[1]


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


def g(m, *ks):
    for k in ks:
        if k in m:
            return m[k]
    return 0.0


def stream(prompt, out, i):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 700, "stream": True,
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
            if not d.get("choices"):
                continue
            c = d["choices"][0].get("delta", {}).get("content")
            if c:
                times.append(time.perf_counter()); text.append(c)
    s = "".join(text)
    out[i] = dict(ttft=times[0] - t0 if times else None, span=times[-1] - times[0] if len(times) > 1 else None,
                  chunks=len(times), chars=len(s), sha=hashlib.sha256(s.encode()).hexdigest()[:16])


def run(prompts, conc):
    m0 = metrics(); out = [None] * len(prompts)
    if conc == 1:
        for i, p in enumerate(prompts):
            stream(p, out, i)
    else:
        th = [threading.Thread(target=stream, args=(p, out, i)) for i, p in enumerate(prompts)]
        t0 = time.perf_counter()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.perf_counter() - t0
    m1 = metrics()
    gen = g(m1, "vllm:generation_tokens_total") - g(m0, "vllm:generation_tokens_total")
    D = g(m1, "vllm:spec_decode_num_drafts_total") - g(m0, "vllm:spec_decode_num_drafts_total")
    DT = g(m1, "vllm:spec_decode_num_draft_tokens_total") - g(m0, "vllm:spec_decode_num_draft_tokens_total")
    A = g(m1, "vllm:spec_decode_num_accepted_tokens_total") - g(m0, "vllm:spec_decode_num_accepted_tokens_total")
    res = dict(gen_tokens=gen, accept_rate=round(100 * A / DT, 2) if DT else None,
               accept_len=round(1 + A / D, 3) if D else None, shas=[o["sha"] for o in out])
    spans = sum(o["span"] for o in out); toks_per_req = gen / len(prompts)
    if conc == 1:
        # decode ms/tok per request: span over (tokens - 1); tokens per request from the chunk-independent counter
        res["decode_ms_per_tok"] = round(1000 * spans / max(1.0, gen - len(prompts)), 3)
    else:
        res["agg_tok_s"] = round(gen / wall, 2); res["wall_s"] = round(wall, 3)
    return res


import os
THR_FILE = "/opt/llm/runners/dyndraft/thr"
SEQ = [0.0, 0.3, 0.5, 0.7, 0.7, 0.5, 0.3, 0.0]


def set_thr(t):
    with open(THR_FILE, "w") as f:
        f.write(f"{t}\n")
    time.sleep(0.5)


set_thr(0.0)
run(PROMPTS[:1], 1)                                  # warm-up, discarded
res = {"sweep": []}
for i, t in enumerate(SEQ):
    set_thr(t)
    run(PROMPTS[:1], 1)                              # one request so the new threshold is live before timing
    c1 = run(PROMPTS, 1)
    c4 = run(PROMPTS, 4)
    res["sweep"].append(dict(round=i // 4, thr=t, c1_ms_tok=c1.get("decode_ms_per_tok"), c1_acc=c1.get("accept_len"),
                             c1_rate=c1.get("accept_rate"), c1_shas=c1["shas"], c4_tok_s=c4.get("agg_tok_s"),
                             c4_acc=c4.get("accept_len"), c4_shas=c4["shas"]))
    print(json.dumps(res["sweep"][-1]), file=sys.stderr, flush=True)
set_thr(0.0)
try:
    lg = open(f"/opt/llm/armrun-{tag}.log", errors="replace").read()
    res["fndyn_active"] = "FNDYN active" in lg
    res["stops_lines"] = [l[-200:] for l in lg.splitlines() if "FNDYN stops" in l][-3:]
except FileNotFoundError:
    res["fndyn_active"] = None
for t in sorted(set(SEQ)):
    rows = [r for r in res["sweep"] if r["thr"] == t]
    res[f"c1_{t}"] = [r["c1_ms_tok"] for r in rows]
print(json.dumps(res))
