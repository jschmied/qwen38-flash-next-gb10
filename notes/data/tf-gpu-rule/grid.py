import json, sys, urllib.request, hashlib
base, out = sys.argv[1], sys.argv[2]
prompts = ["Write a Python function that merges two sorted lists, with a docstring and three tests.",
           "Explain how a refrigerator moves heat out of its cabinet, step by step."]
settings = [dict(temperature=1.0, top_p=1.0, top_k=0), dict(temperature=0.6, top_p=0.8, top_k=20),
            dict(temperature=1.2, top_p=0.95, top_k=50), dict(temperature=1.0, top_p=1.0, top_k=5, min_p=0.05),
            dict(temperature=0.8, top_p=0.9, top_k=0, min_p=0.05)]
rows = []
def ask(body):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
for pi, p in enumerate(prompts):
    for draft in (True, False):
        r = ask({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 96, "temperature": 0, "draft": draft, "logprobs": True, "top_logprobs": 5})
        lp = hashlib.sha256(json.dumps(r["choices"][0].get("logprobs"), sort_keys=True).encode()).hexdigest()[:12]
        rows.append({"prompt": pi, "greedy": True, "draft": draft, "sha": r.get("token_sha") or r.get("usage", {}).get("token_sha"), "text": hashlib.sha256(r["choices"][0]["message"]["content"].encode()).hexdigest()[:12], "logprobs": lp})
        for si, s in enumerate(settings):
            for seed in (1, 7, 42, 1234):
                r = ask({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 96, "seed": seed, "draft": draft, **s})
                rows.append({"prompt": pi, "setting": si, "seed": seed, "draft": draft, "sha": r.get("token_sha") or r.get("usage", {}).get("token_sha"), "text": hashlib.sha256(r["choices"][0]["message"]["content"].encode()).hexdigest()[:12]})
json.dump(rows, open(out, "w"), indent=0)
print(len(rows), "replies;", "token_sha present" if rows[0]["sha"] else "no token_sha field (text hashes only)")
