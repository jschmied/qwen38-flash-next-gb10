"""Long-horizon comparison: per-document mean |dlogprob| by position quarter, and total NLL change, vs base."""
import json, sys, statistics as st
R = "/opt/llm/runners/results"
base = json.load(open(f"{R}/lp-{sys.argv[1]}.json"))
for arm in sys.argv[2:]:
    a = json.load(open(f"{R}/lp-{arm}.json"))
    for doc in sorted(base):
        x, y = base[doc], a[doc]; n = min(len(x), len(y))
        pairs = [(x[i], y[i]) for i in range(n) if x[i] is not None and y[i] is not None]
        q = len(pairs) // 4
        quart = [st.mean(abs(p - r) for p, r in pairs[k * q:(k + 1) * q]) for k in range(4)]
        nll_b = -sum(p for p, _ in pairs); nll_a = -sum(r for _, r in pairs)
        print(f"{arm:10s} {doc}: {len(pairs):5d} tok | mean|dlp| by quarter " + " ".join(f"{v:.4f}" for v in quart)
              + f" | NLL {nll_b/len(pairs):.4f} -> {nll_a/len(pairs):.4f} ({100*(nll_a-nll_b)/nll_b:+.2f} %)")
