#!/usr/bin/env python3
"""Send prompt.txt once to the server on :8092 (prompt $GEN_PROMPT or prompt.txt; seed $GEN_SEED, default 1; the model's default sampling and thinking, 100k max output),
save the reply's content as <out>.html, the full JSON as <out>.json, and print timings. argv: <out-prefix> [key]"""
import json, os, sys, time, urllib.request
out = sys.argv[1]; key = sys.argv[2] if len(sys.argv) > 2 else "none"
prompt = open(os.environ.get("GEN_PROMPT") or __file__.rsplit("/", 1)[0] + "/prompt.txt").read()
b = {"model": "flashnext", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100000,
     "seed": int(os.environ.get("GEN_SEED", "1"))}
t = time.time()
r = urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", json.dumps(b).encode(),
                           {"Content-Type": "application/json", "Authorization": "Bearer " + key})
d = json.loads(urllib.request.urlopen(r, timeout=7200).read())
m = d["choices"][0]["message"]
open(out + ".json", "w").write(json.dumps(d, indent=1))
open(out + ".html", "w").write(m.get("content") or "")
print(json.dumps({"seconds": round(time.time() - t, 1), "usage": d.get("usage"), "finish": d["choices"][0]["finish_reason"],
                  "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                  "html_bytes": len((m.get("content") or "").encode())}))
