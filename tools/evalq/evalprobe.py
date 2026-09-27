#!/usr/bin/env python3
"""Task eval for the GDN-projection precision decision (FP8 prod vs NVFP4 W4A16). Thinking off, greedy, c=16:
GSM8K test (1319, scored here: '#### n' line, else last number) and HumanEval (164, generations saved and scored
offline by evalscore.py as an unprivileged user, never executed here). Plus TTFT at ~8k / ~30k tokens with a
per-request nonce so the prefix cache cannot serve it. Prints ONE json object (armrun contract). argv: <tag>"""
import gzip, json, re, statistics, sys, threading, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; D = "/opt/llm/runners/evalq"
tag = sys.argv[1]; OUT = f"/opt/llm/runners/results/evalq-{tag}.jsonl"; lock = threading.Lock()


def chat(content, max_tokens):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": content}], "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(b).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    c = d["choices"][0]
    return c["message"]["content"] or "", c["finish_reason"], d["usage"]["completion_tokens"]


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    ns = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(ns[-1]) if ns else None


def gsm(item):
    i, q, gold = item
    text, fin, n = chat(q + "\n\nSolve step by step. End with a final line of the form '#### <number>'.", 1024)
    ok = num(text) is not None and abs(num(text) - gold) < 1e-6
    with lock, open(OUT, "a") as f:
        f.write(json.dumps({"task": "gsm8k", "i": i, "ok": ok, "finish": fin, "ntok": n, "text": text}) + "\n")
    return ok, fin, n


def he(item):
    i, p = item
    text, fin, n = chat("Complete the following Python function. Reply with the complete function (including the "
                        "imports it needs) in a single ```python code block.\n\n```python\n" + p["prompt"] + "```",
                        1024)
    with lock, open(OUT, "a") as f:
        f.write(json.dumps({"task": "humaneval", "i": i, "task_id": p["task_id"], "finish": fin, "ntok": n,
                            "text": text}) + "\n")
    return fin, n


def ttft():
    unit = ("You are reviewing a large Python service. Here is the module under discussion. "
            "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")
    res = {}
    for reps, name in ((220, "8k"), (860, "30k")):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            chat(f"[{uuid.uuid4()}]\n" + unit * reps + "\nSummarise what handler() does in one sentence.", 1)
            ts.append(time.perf_counter() - t0)
        res[name] = {"median_s": round(statistics.median(ts), 3), "all": [round(x, 3) for x in ts]}
    return res


t_all = time.time()
tt = ttft()
g = [(i, json.loads(l)) for i, l in enumerate(open(f"{D}/gsm8k_test.jsonl")) if l.strip()]
g = [(i, x["question"], float(x["answer"].split("####")[-1].strip().replace(",", ""))) for i, x in g]
h = [(i, json.loads(l)) for i, l in enumerate(gzip.open(f"{D}/HumanEval.jsonl.gz", "rt")) if l.strip()]
t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    gr = list(ex.map(gsm, g))
t1 = time.time()
with ThreadPoolExecutor(16) as ex:
    hr = list(ex.map(he, h))
t2 = time.time()
print(json.dumps({
    "ttft": tt,
    "gsm8k": {"n": len(gr), "acc": round(sum(o for o, _, _ in gr) / len(gr), 4),
              "correct": sum(o for o, _, _ in gr), "truncated": sum(f == "length" for _, f, _ in gr),
              "gen_tokens": sum(n for _, _, n in gr), "s": round(t1 - t0, 1)},
    "humaneval": {"n": len(hr), "truncated": sum(f == "length" for f, _ in hr),
                  "gen_tokens": sum(n for _, n in hr), "s": round(t2 - t1, 1), "file": OUT},
    "total_s": round(time.time() - t_all, 1)}))
