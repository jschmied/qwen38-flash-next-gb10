#!/usr/bin/env python3
"""vllm#58449 server probe: decode speed with MTP n=3, measured on streamed output (first -> last token), at c=1
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


run(PROMPTS[:1], 1)                                # warm-up (compile / autotune paths), discarded
res = {"c1": run(PROMPTS, 1), "c4": run(PROMPTS, 4)}
p = subprocess.run([sys.executable, "/opt/llm/runners/agentloop2.py", tag], capture_output=True, text=True)
m = re.search(r"TOTAL ([\d.]+) s over 8 turns\s+\(([\d.]+) s/turn\)\s+(\d+) tok\s+([\d.]+) ms/tok", p.stdout)
if m:
    res["agent"] = dict(total_s=float(m[1]), s_per_turn=float(m[2]), tokens=int(m[3]), ms_per_tok=float(m[4]))
try:
    lg = open(f"/opt/llm/armrun-{tag}.log", errors="replace").read()
    res["fallback_line"] = "Fused multi-step draft decode is not supported" in lg
except FileNotFoundError:
    res["fallback_line"] = None
# flat numeric keys for armrun's range summary
res["c1_ms_tok"] = res["c1"].get("decode_ms_per_tok"); res["c4_tok_s"] = res["c4"].get("agg_tok_s")
res["c1_acc_len"] = res["c1"].get("accept_len"); res["s_turn"] = res.get("agent", {}).get("s_per_turn")
print(json.dumps(res))
