#!/usr/bin/env python3
"""Decode-time quality probe: greedy 8 prompts x 512 tokens, chosen-token logprob + top-5 per position, saved per arm
to /opt/llm/runners/results/div-<arm>.json; prints one summary json (armrun contract).  argv: <arm tag>"""
import json, sys, urllib.request
URL = "http://127.0.0.1:8092/v1/completions"; H = {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string and returns total seconds. Explain edge cases.\n",
    "Explain step by step how Dijkstra's algorithm works and prove why it is correct for non-negative weights.\n",
    "Translate into German and explain the grammar: 'The committee has postponed the decision until further notice.'\n",
    "A train leaves at 9:40 and travels 245 km at 70 km/h, then 120 km at 90 km/h. When does it arrive? Show work.\n",
    "Write a bash script that finds the 10 largest files under a directory, excluding .git, and prints human sizes.\n",
    "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.\n",
    "Refactor this code for readability and explain each change:\ndef f(a):\n  r=[]\n  for i in range(len(a)):\n    if a[i]%2==0: r.append(a[i]*a[i])\n  return r\n",
    "Describe how a transformer decoder generates text, including the KV cache, in plain language.\n",
]
tag = sys.argv[1]; out = []
for p in PROMPTS:
    b = {"model": "flashnext", "prompt": p, "max_tokens": 512, "temperature": 0, "logprobs": 5}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), H), timeout=900).read())
    lp = r["choices"][0]["logprobs"]
    out.append({"tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"]})
json.dump(out, open(f"/opt/llm/runners/results/div-{tag}.json", "w"))
print(json.dumps({"prompts": len(out), "tokens": sum(len(o["tokens"]) for o in out), "file": f"div-{tag}.json"}))
