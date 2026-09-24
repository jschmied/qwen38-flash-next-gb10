#!/usr/bin/env python3
"""vllm#53912 on our prod (Flash-Next, MTP n=3, align, prefix cache, disable_eagle_block_drop).
Adapted from Suppressor72's v4 repro (gist 8a8090d4): regex constraints are ignored on our build, so the
high-acceptance arm continues a DESCENDING number sequence greedily and the low-acceptance arm samples random
numbers at temperature 1.0; ignore_eos keeps A at full length; acceptance from /metrics deltas.
Per arm: A = generate 2400 tokens after a ~4k-token prefix. Then ctx = prefix + first ~470 groups of A (cache hit
reaches into decode-written blocks). B x5 = greedy 20 tokens, cache read. R x3 = same with prompt_logprobs=1 (cache
skipped; its cached_tokens is recorded to prove it). Divergent = B != R."""
import json, random, sys, time, urllib.request
BASE = "http://127.0.0.1:8092"; KEY = "sk-bench"; MODEL = "flashnext"


def post(path, payload):
    r = urllib.request.Request(BASE + path, json.dumps(payload).encode(),
                               {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def comp(prompt, max_tokens=20, temperature=0.0, **kw):
    b = post("/v1/completions", dict(model=MODEL, prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kw))
    u = b.get("usage", {})
    return b["choices"][0]["text"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", -1), u.get("prompt_tokens", -1)


def metrics():
    m = {}
    for ln in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        if (ln.startswith("vllm:spec_decode_num_") or ln.startswith("vllm:prefix_cache_")) and " " in ln:
            k, v = ln.rsplit(" ", 1); k = k.split("{")[0]; m[k] = m.get(k, 0.0) + float(v)
    return m


def run_arm(name, prefix, temperature, seed):
    m0 = metrics()
    a, ac, apt = comp(prefix, max_tokens=2400, temperature=temperature, ignore_eos=True, seed=seed)
    m1 = metrics()
    dt = m1["vllm:spec_decode_num_draft_tokens_total"] - m0["vllm:spec_decode_num_draft_tokens_total"]
    acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
    groups = a.split(" ")
    ctx = prefix + " " + " ".join(groups[:470])
    def hit(fn):
        h0 = metrics().get("vllm:prefix_cache_hits_total", 0.0); r = fn(); h1 = metrics().get("vllm:prefix_cache_hits_total", 0.0)
        return r + (h1 - h0,)
    B = [hit(lambda: comp(ctx)) for _ in range(5)]
    R = [hit(lambda: comp(ctx, prompt_logprobs=1)) for _ in range(3)]
    r0 = R[0][0]
    res = dict(arm=name, seed=seed, a_prompt=apt, a_chars=len(a), accept_rate=round(100 * acc / dt, 1) if dt else None,
               b_hits=[int(b[3]) for b in B], ctx_tokens=B[-1][2], r_hits=[int(r[3]) for r in R],
               b_div=sum(1 for b in B if b[0] != r0), b_distinct=len({b[0] for b in B}), r_consistent=len({r[0] for r in R}) == 1,
               b0=B[0][0][:60], r0=r0[:60])
    res["covers_decode_written"] = max(res["b_hits"]) > apt
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


out = []
for seed in (202, 303, 404, 505):
    rng = random.Random(seed)
    start = rng.randrange(8 * 10**6, 9 * 10**6)
    desc = " ".join(str(start - 3 * i) for i in range(500))                      # high-acceptance arm
    rnd = " ".join(str(rng.randrange(10**6, 10**7)) for _ in range(500))         # low-acceptance arm
    out.append(run_arm("countdown-greedy", desc, 0.0, seed))
    out.append(run_arm("random-t1", rnd, 1.0, seed))
print("SUMMARY", json.dumps([(r["arm"], r["seed"], r["accept_rate"], r["b_hits"], r["r_hits"], r["b_div"]) for r in out]))
