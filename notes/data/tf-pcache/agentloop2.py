#!/usr/bin/env python3
"""Agent-loop TTFT probe: a fixed ~6k-token system prompt, then N turns that each append the model's reply and a
~700-token tool result. Streams each request and records TTFT, prompt tokens and any cached-token count."""
import json, sys, time, urllib.request, hashlib
url, label, turns = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = open("/home/jschmied/git/qwen38-flash-next-gb10/notes/kolibri-drafter-plan.md").read()
other = open("/home/jschmied/git/qwen38-flash-next-gb10/notes/upstream/README.md").read()
system = "You are a coding agent. Follow the project notes below exactly.\n\n" + src[:int(sys.argv[4]) if len(sys.argv) > 4 else 22000]
msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Read the notes, then list the open tasks."}]
out = []
for t in range(turns):
    body = json.dumps({"model": "m", "messages": msgs, "max_tokens": 96, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content") or ""
                if c and first is None: first = time.perf_counter() - t0
                text += c
    total = time.perf_counter() - t0
    pt = (usage or {}).get("prompt_tokens"); cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    rec = {"label": label, "turn": t, "prompt_tokens": pt, "cached": cached, "ttft": round(first or total, 3), "total": round(total, 3),
           "reply_sha": hashlib.sha256(text.encode()).hexdigest()[:12]}
    print(json.dumps(rec), flush=True); out.append(rec)
    msgs.append({"role": "assistant", "content": text})
    chunk = other[t * (int(sys.argv[5]) if len(sys.argv) > 5 else 2800):(t + 1) * (int(sys.argv[5]) if len(sys.argv) > 5 else 2800)]
    msgs.append({"role": "user", "content": "Tool output:\n" + chunk + "\nContinue with the next step."})
