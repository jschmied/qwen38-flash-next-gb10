#!/usr/bin/env python3
"""decblk4 (2026-09-28): v1's exact-token-id method with a ~3.1k prompt so the follow-up (~3.55k) crosses the 3456-token
scheduler block inside the generated ids; v2/v3 re-tokenized the generated text and could not match there.
Decode-written cache block probe (vllm#53912 design by Suppressor72, adapted). Per seed: a ~1.4k-token prompt of
random 7-digit numbers, a sampled continuation (temperature 1.0: low draft acceptance, many rejected drafts) that
crosses the 1728-token block boundary, then a follow-up = prompt ids + the first 480 generated ids. B reads the
decode-written cached block; R skips the cache (prompt_logprobs=1). Greedy 24 tokens each, compared token by token.
Prints ONE json object (armrun contract). argv: <tag>"""
import json, random, sys, urllib.request
URL = "http://127.0.0.1:8092"; H = {"Content-Type": "application/json", "Authorization": "Bearer none"}; tag = sys.argv[1]
def post(path, body):
    return json.loads(urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), H), timeout=900).read())
def hits():
    t = 0.0
    for ln in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if ln.startswith("vllm:prefix_cache_hits_total"):
            t += float(ln.rsplit(" ", 1)[1])
    return t
def comp(prompt, n, **kw):
    b = {"model": "flashnext", "prompt": prompt, "max_tokens": n, "return_token_ids": True}; b.update(kw)
    h0 = hits(); d = post("/v1/completions", b); c = d["choices"][0]
    return c["prompt_token_ids"] if "prompt_token_ids" in c else d.get("prompt_token_ids"), c["token_ids"], hits() - h0
rows = []
for s in range(1, 17):
    rnd = random.Random(s)
    text = "Numbers:\n" + " ".join(str(rnd.randint(1000000, 9999999)) for _ in range(380)) + "\nContinue the list:"
    pids, gids, _ = comp(text, 520, temperature=1.0, seed=s, ignore_eos=True)
    follow = list(pids) + list(gids[:480])
    _, b, hb = comp(follow, 24, temperature=0)
    _, r, hr = comp(follow, 24, temperature=0, prompt_logprobs=1)
    fd = next((i for i, (x, y) in enumerate(zip(b, r)) if x != y), None)
    rows.append({"seed": s, "prompt": len(pids), "follow": len(follow), "hits_B": hb, "hits_R": hr,
                 "equal": b == r, "first_div": fd})
print(json.dumps({"n": len(rows), "divergent": sum(not x["equal"] for x in rows),
                  "B_hit_past_prompt": all(x["hits_B"] > x["prompt"] for x in rows), "B_hits": sorted(set(x["hits_B"] for x in rows)), "R_miss_all": all(x["hits_R"] == 0 for x in rows),
                  "prompt_len": [rows[0]["prompt"], rows[-1]["prompt"]], "rows": rows}))
