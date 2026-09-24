#!/usr/bin/env python3
"""Whole-step kernel profile of steady-state c=1 MTP decode (speed-of-light step 3). Warm up, measure decode ms/tok
unprofiled, then profile one ~150-token essay request between /start_profile and /stop_profile. Prints ONE json."""
import glob, json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8092"; H = {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}
PDIR = "/opt/llm/capture/prof-0925"
P = "Explain in about 500 words how a B-tree keeps itself balanced during inserts and deletes."
def post(path, body=None, timeout=1800):
    r = urllib.request.Request(URL + path, json.dumps(body or {}).encode(), H)
    return urllib.request.urlopen(r, timeout=timeout).read()
def decode(max_tokens):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": max_tokens, "stream": True,
         "messages": [{"role": "user", "content": P}], "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(), H)
    ts = []
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            ln = raw.decode().strip()
            if ln.startswith("data:") and ln != "data: [DONE]":
                d = json.loads(ln[5:])
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    ts.append(time.perf_counter())
    return round(1000 * (ts[-1] - ts[0]) / max(1, len(ts) - 1), 3), len(ts)
decode(64); decode(64)                                   # warm-up
base = decode(400)
post("/start_profile"); t0 = time.time()
prof = decode(150)
post("/stop_profile"); t_stop = round(time.time() - t0, 1)
files = sorted(glob.glob(os.path.join(PDIR, "**", "*.json*"), recursive=True))
print(json.dumps({"base_ms_per_chunk": base[0], "base_chunks": base[1], "prof_ms_per_chunk": prof[0],
                  "prof_chunks": prof[1], "stop_s": t_stop, "trace_files": len(files),
                  "trace_mb": round(sum(os.path.getsize(f) for f in files) / 2**20, 1)}))
