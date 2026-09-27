#!/usr/bin/env python3
"""Load-time probe: weight-load seconds from this arm's server log (armrun names it /opt/llm/armrun-<tag>.log), plus one
greedy request's hash as a sanity check that the patched loader loads the same weights. ONE json. argv: <tag>"""
import hashlib, json, re, sys, urllib.request
tag = sys.argv[1]; log = open(f"/opt/llm/armrun-{tag}.log", errors="replace").read()
loads = [float(x) for x in re.findall(r"Loading weights took ([0-9.]+) seconds", log)]
total = re.findall(r"Model loading took [0-9.]+ GiB memory and ([0-9.]+) seconds", log)
b = {"model": "flashnext", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists."}]}
r = urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", json.dumps(b).encode(),
                           {"Content-Type": "application/json", "Authorization": "Bearer none"})
txt = json.loads(urllib.request.urlopen(r, timeout=600).read())["choices"][0]["message"]["content"]
print(json.dumps({"weights_s": loads, "model_loading_s": [float(x) for x in total],
                  "greedy_sha": hashlib.sha256(txt.encode()).hexdigest()[:16]}))
