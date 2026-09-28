#!/usr/bin/env python3
"""F4 follow-up: the §5z 8k-prompt + 96-token request (nvprobe's replay) paid +0.35..0.5 s with full graphs. This probe
repeats it with streaming, token counts, finish reason and output hash, in three phases on one server: replay pairs on
the fresh server (A), then nvprobe's TTFT probes (8k x3, 30k x3), then replay pairs again (B). Fixed nonces per phase and
pair, so arms compare. Prints ONE json. argv: <tag>"""
import hashlib, json, sys, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
unit = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")


def chat(content, n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "stream": True,
         "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); first = last = None; text = []; usage = None; fin = None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                fin = c.get("finish_reason") or fin
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter(); first = first or now; last = now; text.append(piece)
    end = time.perf_counter(); ct = usage["completion_tokens"]
    return {"total_s": round(end - t0, 3), "ttft_s": round(first - t0, 3), "tokens": ct, "finish": fin,
            "decode_ms_tok": round(1000 * (last - first) / max(1, ct - 1), 3),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}


def replay(phase):
    rows = []
    for i in range(3):
        q = f"[replay-{phase}-{i}]\n" + unit * 220 + "\nList three risks in this module, one line each."
        rows.append({"cold": chat(q, 96), "warm": chat(q, 96)})
    return rows


out = {"A": replay("A")}
tt = {}
for reps, name in ((220, "8k"), (860, "30k")):
    tt[name] = [chat(f"[ttft-{name}-{i}]\n" + unit * reps + "\nSummarise what handler() does in one sentence.", 1)["ttft_s"]
                for i in range(3)]
out["ttft"] = tt
out["B"] = replay("B")
print(json.dumps(out))
