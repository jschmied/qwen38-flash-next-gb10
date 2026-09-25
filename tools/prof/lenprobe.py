import json, time, urllib.request
URL = "http://127.0.0.1:8092"; H = {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}
P = "Explain in about 500 words how a B-tree keeps itself balanced during inserts and deletes."
def decode(n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "stream": True, "messages": [{"role": "user", "content": P}],
         "chat_template_kwargs": {"enable_thinking": False}}
    ts = []
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(), H), timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if ln.startswith("data:") and ln != "data: [DONE]":
                d = json.loads(ln[5:])
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"): ts.append(time.perf_counter())
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    first150 = sum(gaps[:53]) / 53 * 1000 if len(gaps) >= 53 else None
    return round(1000 * (ts[-1] - ts[0]) / (len(ts) - 1), 2), len(ts), round(first150, 2) if first150 else None
decode(64); decode(64)
for r in range(3):
    print("decode150", decode(150), "| decode400", decode(400), flush=True)
