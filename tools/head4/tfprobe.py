#!/usr/bin/env python3
"""Teacher-forced head probe (2026-09-30, FNHEAD4 screen). Reference text, not model output: the first 40 HumanEval
problems (prompt + canonical solution) and the first 40 GSM8K test items (question + reference answer). For each,
/v1/completions with prompt_logprobs=1 and return_token_ids returns every reference token's logprob and rank under the served model; we report
mean NLL per token and top-1 accuracy (rank 1) per set, plus a hash of the rank vector (identical arms => identical
hash). The cache is salted per request, so every prompt is computed cold. Prints ONE json. argv: <tag>"""
import gzip, hashlib, json, sys, urllib.request, uuid
URL = "http://127.0.0.1:8092/v1/completions"; KEY = "sk-bench"; D = "/opt/llm/runners/evalq"; tag = sys.argv[1]


def score(text):
    b = {"model": "flashnext", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
         "return_token_ids": True, "cache_salt": uuid.uuid4().hex}
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    c = d["choices"][0]
    lp = c.get("prompt_logprobs") or d.get("prompt_logprobs")
    ids = c.get("prompt_token_ids") or d.get("prompt_token_ids")
    out = []
    for tid, pos in zip(ids[1:], lp[1:]):
        v = pos[str(tid)] if str(tid) in pos else pos[tid]  # the reference token's entry, whatever its rank
        out.append((float(v["logprob"]), int(v["rank"])))
    return out


he = [json.loads(l) for l in gzip.open(f"{D}/HumanEval.jsonl.gz", "rt")][:40]
gs = [json.loads(l) for l in open(f"{D}/gsm8k_test.jsonl")][:40]
sets = {"humaneval": [p["prompt"] + p["canonical_solution"] for p in he],
        "gsm8k": [f"Question: {p['question']}\nAnswer: {p['answer']}" for p in gs]}
res = {"tag": tag}
for name, texts in sets.items():
    toks = [t for x in texts for t in score(x)]
    ranks = [r for _, r in toks]
    res[name] = {"tokens": len(toks), "nll": round(-sum(l for l, _ in toks) / len(toks), 5),
                 "top1": round(sum(r == 1 for r in ranks) / len(ranks), 5),
                 "rank_hash": hashlib.sha256(json.dumps(ranks).encode()).hexdigest()[:16]}
print(json.dumps(res))
