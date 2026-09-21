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
if err:
    print(f"  {err}")
t0 = time.time(); toks = 0
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
        print("  first reply:", repr(d["choices"][0]["message"]["content"][:140]))
dt = time.time() - t0
after, _ = metrics()

print(f"  {a.n} requests, {toks} completion tokens, {dt:.1f}s -> {toks/dt:.2f} tok/s aggregate")
if after:
    acc = draft = 0.0
    for k, v in after.items():
        d = v - before.get(k, 0.0)
        if "accepted" in k: acc += d
        elif "draft" in k or "num_drafts" in k: draft += d
        print(f"    {k}: +{d:.0f}")
    if draft:
        print(f"  acceptance rate: {acc/draft:.3f}   (accepted {acc:.0f} / drafted {draft:.0f})")
        print(f"  acceptance length: {1 + acc/max(draft/ max(1,acc/max(acc,1)),1):.2f} (see raw counters above)")
else:
    print("  no spec_decode counters exposed; compare tok/s against the BF16 drafter instead")
