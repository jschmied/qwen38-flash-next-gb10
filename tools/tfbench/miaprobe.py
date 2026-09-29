#!/usr/bin/env python3
"""MiaAI-Lab's TensorFold decode benchmark (Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold @ 4cd99569, tools/bench.py)
against our server: the same two prompts and settings — "Write a Python quicksort with docstring and tests." at
temperature 0 and "Explain why the sky is blue in a few paragraphs." at the server's default sampling, 256 tokens,
median of 5, thinking left at the template default. Their decode rate comes from TensorFold's server stats, which vLLM
does not emit, so this reports the client-side equivalent: (completion tokens - 1) / (last token - first token), with
reasoning and content tokens both counted as their client does. Prints ONE json. argv: tag"""
import json, statistics, sys, time, urllib.request

URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; tag = sys.argv[1]


def run(prompt, max_tokens, temperature=None):
    body = {"model": "flashnext", "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    start, first, last, usage = time.time(), None, None, {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                now = time.time()
                first = first if first is not None else now
                last = now
            if chunk.get("usage"):
                usage = chunk["usage"]
    n = usage.get("completion_tokens", 0)
    return {"ttft": (first or time.time()) - start, "tokens": n,
            "decode_tps": (n - 1) / (last - first) if first and last and last > first and n > 1 else None}


out = {"tag": tag}
for name, prompt, temp in (("code_greedy", "Write a Python quicksort with docstring and tests.", 0),
                           ("chat_sampled", "Explain why the sky is blue in a few paragraphs.", None)):
    run(prompt, 256, temp)  # warm (their client does not; our first request after a start is in the cold window)
    rs = [run(prompt, 256, temp) for _ in range(5)]
    tps = [r["decode_tps"] for r in rs if r["decode_tps"]]
    out[name] = {"decode_tps_median": round(statistics.median(tps), 1), "all": [round(x, 1) for x in tps],
                 "tokens": [r["tokens"] for r in rs], "ttft_median": round(statistics.median(r["ttft"] for r in rs), 3)}
print(json.dumps(out))
