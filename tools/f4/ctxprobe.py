#!/usr/bin/env python3
"""Decode speed vs context length (F4 follow-up). For each context size: a unique-nonce prompt (no prefix hit within a server; fixed per size and request, so arms compare), streamed,
greedy, 256 tokens, ignore_eos. decode ms/tok = (last chunk - first content chunk) / (completion_tokens - 1), so prefill
is excluded. 2 requests per size. Also the output hash per size (both arms must match). Prints ONE json. argv: <tag>"""
import hashlib, json, sys, time, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
unit = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")


def run(reps, tag, n=256):
    q = f"[ctx-{tag}]\n" + unit * reps + "\nExplain step by step what this module does and how to test it."
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "ignore_eos": True, "stream": True,
         "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": q}],
         "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); first = last = None; text = []; usage = None
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter(); first = first or now; last = now; text.append(piece)
    ct = usage["completion_tokens"]
    return {"prompt": usage["prompt_tokens"], "ttft_s": round(first - t0, 3),
            "decode_ms_tok": round(1000 * (last - first) / (ct - 1), 3), "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}


out = {}
for reps, name in ((30, "1k"), (220, "8k"), (470, "16k"), (820, "28k")):
    rs = [run(reps, f"{name}-{i}") for i in range(2)]
    out[name] = {"prompt": rs[0]["prompt"], "decode_ms_tok": [x["decode_ms_tok"] for x in rs],
                 "ttft_s": [x["ttft_s"] for x in rs], "sha": [x["sha"] for x in rs]}
print(json.dumps(out))
