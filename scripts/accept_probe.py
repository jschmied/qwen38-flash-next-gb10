#!/usr/bin/env python3
"""Acceptance length for an MTP drafter, from the engine's own spec-decode counters.

This is the measurement that exposes a bad drafter: a mis-scaled draft still emits fluent tokens,
the target rejects them, and only acceptance moves. Throughput alone would half-hide it.

Counters are read before and after a fixed workload, so the value is for THIS workload, not the
server's lifetime.
"""
import argparse, json, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8092")
ap.add_argument("--arm", default="?")   # armrun passes this; without it the probe dies
ap.add_argument("--key", default="sk-bench")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=160)
a = ap.parse_args()

def metrics():
    try:
        raw = urllib.request.urlopen(f"{a.url}/metrics", timeout=20).read().decode()
    except Exception as e:
        return {}, f"metrics unavailable: {e}"
    out = {}
    for ln in raw.splitlines():
        if ln.startswith("#") or "spec_decode" not in ln:
            continue
        m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", ln)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out, None

PROMPTS = [
    "Explain why mixture-of-experts models need a router, in three sentences.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "Summarise the tradeoff between latency and throughput in LLM serving.",
    "What is speculative decoding and why does acceptance length matter?",
    "Describe the difference between prefill and decode in one paragraph.",
    "List five practical uses of quantization in model serving.",
]

before, err = metrics()
t0 = time.time(); toks = 0; first = ""
for i in range(a.n):
    body = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": a.max_tokens,
                       "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(f"{a.url}/v1/chat/completions", body,
                               {"Content-Type": "application/json",
                                "Authorization": "Bearer " + a.key})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    toks += d["usage"]["completion_tokens"]
    if i == 0:
        first = d["choices"][0]["message"]["content"][:90]
dt = time.time() - t0
after, _ = metrics()

out = {"arm": a.arm, "requests": a.n, "completion_tokens": toks,
       "seconds": round(dt, 2), "tok_s": round(toks / dt, 2), "first_reply": first}
# Use the three exact counters. Summing every key containing "accepted" double-counts
# num_accepted_tokens_per_pos_total and yields a rate > 1 (measured 1.1983 once) and an
# acceptance length above the n+1 ceiling.
dr = after.get("vllm:spec_decode_num_drafts_total", 0.0) - before.get("vllm:spec_decode_num_drafts_total", 0.0)
dt = after.get("vllm:spec_decode_num_draft_tokens_total", 0.0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
ac = after.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
for k, v in after.items():
    out[f"m_{k.split(':')[-1]}"] = round(v - before.get(k, 0.0), 1)
out["drafts"], out["draft_tokens"], out["accepted"] = dr, dt, ac
if dt > 0:
    out["acceptance_rate"] = round(ac / dt, 4)
if dr > 0:
    out["acceptance_length"] = round(1.0 + ac / dr, 3)
    out["spec_n"] = round(dt / dr, 2)
    # a rate above 1, or a length above n+1, means the counters were mis-aggregated
    if out.get("acceptance_rate", 0) > 1.0 or out["acceptance_length"] > out["spec_n"] + 1.001:
        out["WARNING"] = "derived values exceed their ceiling -- counter aggregation is wrong"
print(json.dumps(out), flush=True)
