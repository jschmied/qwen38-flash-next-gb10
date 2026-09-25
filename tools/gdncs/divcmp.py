"""Compare decode-time divergence of arms vs base: first divergent token, |d logprob| before it, top-5 overlap."""
import json, sys, statistics as st
R = "/opt/llm/runners/results"
base = json.load(open(f"{R}/div-{sys.argv[1]}.json"))
for arm in sys.argv[2:]:
    a = json.load(open(f"{R}/div-{arm}.json")); first = []; dlp = []; top_overlap = []; mism_top1 = 0; npos = 0
    for x, y in zip(base, a):
        n = min(len(x["tokens"]), len(y["tokens"])); k = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), n)
        first.append(k)
        for i in range(k):
            if x["token_logprobs"][i] is not None and y["token_logprobs"][i] is not None:
                dlp.append(abs(x["token_logprobs"][i] - y["token_logprobs"][i]))
            tx, ty = set((x["top"][i] or {}).keys()), set((y["top"][i] or {}).keys())
            if tx: top_overlap.append(len(tx & ty) / len(tx))
            npos += 1
    q = sorted(dlp)
    print(f"{arm:10s} first divergence per prompt {first}  median {st.median(first)} | before it: |dlogprob| mean {st.mean(dlp):.4f} p99 {q[int(.99*len(q))]:.4f} max {q[-1]:.4f} | top-5 overlap {st.mean(top_overlap):.3f} ({npos} positions)")
