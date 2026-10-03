"""Kolibri on TensorFold: greedy replies alone vs four at once (must be equal), aggregate decode, a long-context needle."""

import json
import sys
import threading
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8093/v1/chat/completions"
QS = ["Erkläre in drei Sätzen, was ein Transformer ist.", "Write a haiku about the Rhine.",
      "Nenne fünf deutsche Dichter und je ein Werk.", "What is 2^20? Explain briefly."]


def ask(content, max_tokens=256, effort="none"):
    body = {"model": "kolibri-1", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": effort}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["choices"][0]["message"].get("content") or "", r["usage"], time.perf_counter() - t0


solo = [ask(q) for q in QS]
out = [None] * len(QS)
t0 = time.perf_counter()
ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, ask(QS[i]))) for i in range(len(QS))]
[t.start() for t in ts]
[t.join() for t in ts]
wall = time.perf_counter() - t0
same = [a[0] == b[0] for a, b in zip(solo, out)]
toks = sum(o[1]["completion_tokens"] for o in out)
print("concurrent == solo:", same)
print(f"solo tok/s: {[round(s[1]['completion_tokens'] / s[2], 1) for s in solo]}")
print(f"4 at once: {toks} tokens in {wall:.1f}s = {toks / wall:.1f} tok/s aggregate")
for q, s in zip(QS, solo):
    print("Q:", q, "\nA:", s[0][:160].replace("\n", " "))
# long context: a needle in ~N tokens of filler
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
filler = "Der Himmel ist blau und das Gras ist grün. " * (n // 11)
mid = len(filler) // 2
doc = filler[:mid] + " Die geheime Zahl lautet 48151623. " + filler[mid:]
a, u, dt = ask(doc + "\n\nWie lautet die geheime Zahl? Antworte nur mit der Zahl.", 32)
print(f"needle: {u['prompt_tokens']} prompt tokens, {dt:.1f}s, answer {a!r}")
