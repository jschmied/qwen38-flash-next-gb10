#!/usr/bin/env python3
"""Warm-turn TTFT probe for the vllm#54458 / prompt-end-state A/B (armrun contract: ONE json object). Adapted from
turnfit.py (finding 141). (A) 20k cached prefix + N fresh tokens, 3 reps. (B) replay of 2 held-out SWE-bench
trajectories flattened into one growing user message (pure prefix extension, the agent-loop case), max_tokens=1.
Recomputed tokens per turn = prompt_tokens - prefix-cache hit tokens (metrics delta). argv: <tag>"""
import glob, hashlib, json, random, statistics, sys, time, urllib.request
URL = "http://127.0.0.1:8092"; H = {"Content-Type": "application/json", "Authorization": "Bearer none"}; tag = sys.argv[1]
UNIT = ("You are reviewing a large Python service. Here is the module under discussion. def handler(req):\n"
        "    ctx = build_context(req)\n    return dispatch(ctx)\n")
random.seed(7); W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma parse route cache token".split()
def fresh(n): return " ".join(random.choice(W) + str(random.randint(0, 999)) for _ in range(max(1, int(n * 0.55))))
def hits():
    m = {}
    for ln in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if ln.startswith("#") or " " not in ln: continue
        k, v = ln.rsplit(" ", 1); k = k.split("{")[0]
        try: m[k] = m.get(k, 0.0) + float(v)
        except ValueError: pass
    return m.get("vllm:prefix_cache_hits_total", 0.0)
def chat(content, n=1):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": n, "messages": [{"role": "user", "content": content}],
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    h0 = hits(); t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", b, H), timeout=900).read())
    el = time.perf_counter() - t0
    return el, d["usage"]["prompt_tokens"], hits() - h0, d["choices"][0]["message"]["content"] or ""
out = {}
base = f"[{random.random()}]\n" + UNIT * 590
chat(base + "\nSummarise."); chat(base + "\nSummarise.")
A = {}
for N in (16, 256, 1024):
    ts = []; rc = []
    for _ in range(3):
        el, pt, h, _ = chat(base + "\n" + fresh(N) + "\nSummarise."); ts.append(el); rc.append(pt - h)
    A[str(N)] = {"ttft_median_s": round(statistics.median(ts), 3), "recomputed_median": statistics.median(rc)}
out["A"] = A
fs = sorted(glob.glob('/opt/llm/swebench-runs/**/*.traj.json', recursive=True))
held = [f for f in fs if int(hashlib.md5((json.load(open(f)).get("instance_id", "")).encode()).hexdigest(), 16) % 10 == 0][:2]
turns = []; last_prompt = ""
for f in held:
    m = json.load(open(f))["messages"]; sysmsg = next((x["content"] for x in m if x["role"] == "system"), "")[:1500]
    lines = []; prev = 0; turn = 0
    for x in m[1:]:
        r = x["role"]; c = x.get("content") or ""
        if r == "assistant":
            cmd = ""
            for t in x.get("tool_calls") or []:
                try: cmd = json.loads(t["function"]["arguments"]).get("command", "")
                except Exception: cmd = t["function"]["arguments"]
            lines.append(f"ASSISTANT:\n{(x.get('reasoning_content') or '')[:600]}\n```bash\n{cmd}\n```")
        elif r == "tool":
            lines.append(f"TOOL OUTPUT:\n{c[:3000]}"); turn += 1
            p = sysmsg + "\n\nTranscript so far:\n\n" + "\n\n".join(lines) + "\n\nContinue: think through the next step and give the next bash command."
            el, pt, h, _ = chat(p); turns.append({"first": turn == 1, "ttft": el, "pt": pt, "new": pt - prev, "recomputed": pt - h}); prev = pt; last_prompt = p
            if turn >= 24: break
        else: lines.append(f"USER:\n{c}")
warm = [t for t in turns if not t["first"]]  # drop each trajectory's cold first turn
out["B"] = {"turns": len(warm), "ttft_median_s": round(statistics.median(t["ttft"] for t in warm), 3),
            "ttft_mean_s": round(statistics.mean(t["ttft"] for t in warm), 3),
            "new_median": statistics.median(t["new"] for t in warm),
            "recomputed_median": statistics.median(t["recomputed"] for t in warm),
            "recomputed_sum": sum(t["recomputed"] for t in warm), "new_sum": sum(t["new"] for t in warm)}
_, _, _, txt = chat(last_prompt, 64)
out["final_greedy_sha"] = hashlib.sha256(txt.encode()).hexdigest()[:16]
print(json.dumps(out))
