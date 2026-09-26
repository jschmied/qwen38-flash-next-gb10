#!/usr/bin/env python3
"""Code-vs-prose decode probe for the MTP depth sweep. Four code-writing prompts and the four prose prompts of
decprobe, streamed, thinking off: code c=1 greedy, code c=1 sampled (the model's defaults 1.0 / 0.95 / 20, fixed
seeds), prose c=1 greedy, code c=4 greedy. Per cell: decode ms/tok (first -> last streamed token), accept_len and
per-position acceptance from the server counters, output hashes. Prints ONE json object (armrun contract). argv: <tag>"""
import hashlib, json, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; MET = "http://127.0.0.1:8092/metrics"; KEY = "sk-bench"
CODE = [
    "Write a complete Python module implementing an LRU cache with per-entry TTL expiry: type hints, docstrings, "
    "thread safety with a lock, and a pytest test file covering eviction, expiry and concurrent access.",
    "Write a TypeScript Express REST API for a todo list: CRUD endpoints, request validation with zod, an in-memory "
    "repository class, error-handling middleware, and example curl commands.",
    "Write a Rust command-line tool that reads a CSV file and prints per-column statistics (count, mean, min, max, "
    "number of empty cells) using the clap and csv crates, with proper error handling and unit tests.",
    "Implement Dijkstra's shortest-path algorithm in C++17 with a binary heap and an adjacency list, including a "
    "main() that builds a sample graph, prints every shortest distance and reconstructs one path.",
]
PROSE = [
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
        k, v = ln.rsplit(" ", 1)
        name = k.split("{")[0]
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total" and 'position="' in k:
            name += "_" + k.split('position="')[1].split('"')[0]
        try:
            m[name] = m.get(name, 0.0) + float(v)
        except ValueError:
            pass
    return m


def stream(prompt, out, i, sampled, seed):
    body = {"model": "flashnext", "max_tokens": 700, "stream": True, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed} if sampled else {"temperature": 0})
    r = urllib.request.Request(URL, json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); times = []; text = []
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            ln = raw.decode().strip()
            if not ln.startswith("data:") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[5:])
            c = d["choices"][0].get("delta", {}).get("content") if d.get("choices") else None
            if c:
                times.append(time.perf_counter()); text.append(c)
    s = "".join(text)
    out[i] = dict(span=times[-1] - times[0] if len(times) > 1 else 0.0, sha=hashlib.sha256(s.encode()).hexdigest()[:16])


def cell(prompts, conc, sampled=False):
    m0 = metrics(); out = [None] * len(prompts); t0 = time.perf_counter()
    if conc == 1:
        for i, p in enumerate(prompts):
            stream(p, out, i, sampled, 1000 + i)
    else:
        th = [threading.Thread(target=stream, args=(p, out, i, sampled, 1000 + i)) for i, p in enumerate(prompts)]
        [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0; m1 = metrics()
    d = lambda k: m1.get(k, 0.0) - m0.get(k, 0.0)
    gen = d("vllm:generation_tokens_total"); D = d("vllm:spec_decode_num_drafts_total")
    A = d("vllm:spec_decode_num_accepted_tokens_total")
    pos = [round(d(f"vllm:spec_decode_num_accepted_tokens_per_pos_total_{p}") / D, 3) if D else None for p in range(8)
           if f"vllm:spec_decode_num_accepted_tokens_per_pos_total_{p}" in m1]
    res = dict(gen_tokens=gen, accept_len=round(1 + A / D, 3) if D else None, per_pos=pos, shas=[o["sha"] for o in out])
    if conc == 1:
        res["decode_ms_per_tok"] = round(1000 * sum(o["span"] for o in out) / max(1.0, gen - len(prompts)), 3)
    else:
        res["agg_tok_s"] = round(gen / wall, 2)
    return res


stream(CODE[0], [None], 0, False, 0)  # warm pass
out = {"code_c1": cell(CODE, 1), "code_c1_sampled": cell(CODE, 1, True), "prose_c1": cell(PROSE, 1),
       "code_c4": cell(CODE, 4)}
out["code_ms_tok"] = out["code_c1"]["decode_ms_per_tok"]; out["prose_ms_tok"] = out["prose_c1"]["decode_ms_per_tok"]
print(json.dumps(out))
