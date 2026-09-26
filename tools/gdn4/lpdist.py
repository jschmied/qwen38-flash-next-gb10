"""Distance of each arm's teacher-forced prompt logprobs from a reference arm: mean / median / p90 / p99 of |dlogprob|,
share of tokens with |dlp| > 0.5 and > 2, and the NLL change. argv: <ref tag> <arm tag>... (files lp-<tag>.json)."""
import json, sys
R = "/opt/llm/runners/results"
def load(t): return json.load(open(f"{R}/lp-{t}.json"))
ref = load(sys.argv[1])
print(f"{'arm':14s} {'doc':5s} {'tok':>6s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'p99':>7s} {'>0.5':>6s} {'>2':>6s}  NLL ref->arm")
for arm in sys.argv[2:]:
    a = load(arm); allv = []; nb = na = 0
    for doc in sorted(ref):
        pairs = [(x, y) for x, y in zip(ref[doc], a[doc]) if x is not None and y is not None]
        d = sorted(abs(x - y) for x, y in pairs); n = len(d); allv += d
        b = -sum(x for x, _ in pairs) / n; c = -sum(y for _, y in pairs) / n; nb += b * n; na += c * n
        print(f"{arm:14s} {doc:5s} {n:6d} {sum(d)/n:7.4f} {d[n//2]:7.4f} {d[int(n*.9)]:7.4f} {d[int(n*.99)]:7.4f} "
              f"{sum(v > .5 for v in d)/n:6.3f} {sum(v > 2 for v in d)/n:6.3f}  {b:.4f} -> {c:.4f} ({100*(c-b)/b:+.2f} %)")
    allv.sort(); n = len(allv)
    print(f"{arm:14s} {'ALL':5s} {n:6d} {sum(allv)/n:7.4f} {allv[n//2]:7.4f} {allv[int(n*.9)]:7.4f} {allv[int(n*.99)]:7.4f} "
          f"{sum(v > .5 for v in allv)/n:6.3f} {sum(v > 2 for v in allv)/n:6.3f}  NLL {100*(na-nb)/nb:+.2f} %")
