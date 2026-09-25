#!/usr/bin/env python3
"""Long-horizon quality probe: teacher-forced prompt logprobs of long documents (the server runs with a small
max-num-batched-tokens so the GDN state is stored/reloaded every chunk). Saves per-token logprobs per doc.
argv: <arm tag>"""
import glob, json, sys, urllib.request
URL = "http://127.0.0.1:8092/v1/completions"; H = {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}
tag = sys.argv[1]; out = {}
for f in sorted(glob.glob("/opt/llm/runners/gdncs/docs/*.md")):
    text = open(f).read()[:30000]
    b = {"model": "flashnext", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), H), timeout=1800).read())
    pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
    lps = []
    for d in pl[1:]:
        v = list(d.values())[0] if d else None
        lps.append(v["logprob"] if isinstance(v, dict) else v)
    out[f.rsplit("/", 1)[1]] = lps
json.dump(out, open(f"/opt/llm/runners/results/lp-{tag}.json", "w"))
print(json.dumps({"docs": len(out), "tokens": {k: len(v) for k, v in out.items()}}))
