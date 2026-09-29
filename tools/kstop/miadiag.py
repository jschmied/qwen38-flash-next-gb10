#!/usr/bin/env python3
"""Agenda 2d probe (2026-09-29): MiaAI's quicksort prompt alone ("Write a Python quicksort with docstring and tests.",
temperature 0, 256 tokens, thinking at the template default), 5 runs, each with vLLM's spec-decode counter deltas: drafts
(verify cycles), accepted tokens, and acceptance per draft position. Also the output hash and the client-side decode rate
(streamed, as miaprobe). The stop's scheduler counts trimmed drafts as rejected, so per-position acceptance, not
draft_tokens, is comparable across arms. Prints ONE json (armrun contract). argv: tag"""
import hashlib, json, statistics, sys, time, urllib.request

BASE = "http://127.0.0.1:8092"; KEY = "sk-bench"; tag = sys.argv[1]
PROMPT = "Write a Python quicksort with docstring and tests."


def metrics():
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}
    for ln in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        if ln.startswith("#"):
            continue
        v = float(ln.rsplit(" ", 1)[1])
        if ln.startswith("vllm:spec_decode_num_drafts_total"):
            m["drafts"] += v
        elif ln.startswith("vllm:spec_decode_num_draft_tokens_total"):
            m["draft_tokens"] += v
        elif ln.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            m["accepted"] += v
        elif ln.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = int(ln.split('position="')[1].split('"')[0])
            m["pos"][p] = m["pos"].get(p, 0.0) + v
    return m


def run():
    body = {"model": "flashnext", "stream": True, "max_tokens": 256, "temperature": 0,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    first = last = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            d = (chunk.get("choices") or [{}])[0].get("delta", {})
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
            if piece:
                now = time.time(); first = first or now; last = now; text.append(piece)
            if chunk.get("usage"):
                usage = chunk["usage"]
    n = usage.get("completion_tokens", 0)
    return n, (n - 1) / (last - first), hashlib.sha256("".join(text).encode()).hexdigest()[:16]


rows = []
for _ in range(5):
    m0 = metrics(); n, tps, h = run(); m1 = metrics()
    dr = m1["drafts"] - m0["drafts"]; acc = m1["accepted"] - m0["accepted"]
    pos = {p: round((m1["pos"][p] - m0["pos"].get(p, 0.0)) / dr, 3) if dr else None for p in sorted(m1["pos"])}
    rows.append({"tokens": n, "tps": round(tps, 1), "sha": h, "cycles": dr, "accepted_per_cycle": round(acc / dr, 3)
                 if dr else None, "tokens_per_cycle": round(n / dr, 3) if dr else None, "accept_by_pos": pos})
print(json.dumps({"tag": tag, "tps_median": statistics.median(r["tps"] for r in rows),
                  "hashes": sorted(set(r["sha"] for r in rows)), "rows": rows}))
