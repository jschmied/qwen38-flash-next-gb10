#!/usr/bin/env python3
"""Draft-log probe: greedy c=1 over codeprobe's 4 code and 4 prose prompts (700 tokens, thinking off), recording which
lines of $FN_DRAFTLOG each prompt produced, so the offline replay can split code from prose. ONE json (armrun). argv: tag"""
import json, os, sys, urllib.request
import ast, types
_src = ast.parse(open("/opt/llm/runners/ksweep/codeprobe.py").read())
cp = types.SimpleNamespace(URL="http://127.0.0.1:8092/v1/chat/completions", KEY="sk-bench", **{
    n.targets[0].id: ast.literal_eval(n.value) for n in _src.body
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("CODE", "PROSE")})
LOG = "/opt/llm/.cache-armrun/draftlog/draftlog.jsonl"
def lines():
    try: return sum(1 for _ in open(LOG))
    except FileNotFoundError: return 0
seg = []
for kind, prompts in (("code", cp.CODE), ("prose", cp.PROSE)):
    for i, p in enumerate(prompts):
        a = lines()
        b = {"model": "flashnext", "max_tokens": 700, "temperature": 0, "messages": [{"role": "user", "content": p}],
             "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.Request(cp.URL, json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + cp.KEY})
        d = json.loads(urllib.request.urlopen(r, timeout=900).read())
        seg.append({"kind": kind, "i": i, "lines": [a, lines()], "tokens": d["usage"]["completion_tokens"]})
print(json.dumps({"log": LOG, "segments": seg, "total_lines": lines()}))
