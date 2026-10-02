#!/usr/bin/env python3
"""#260 deep profile: kernel time by category and by kernel, GPU busy vs the captured span, from an nsys sqlite export.
argv: <a.sqlite> [<b.sqlite> ...]  (several: side by side, same categories)."""
import collections, re, sqlite3, sys

CATS = [  # first match wins; on the demangled name
    ("dense EXL3 decode-size (linear_kernel)", r"linear_kernel"),
    ("dense EXL3 mid-M (#260)", r"linear_mpg|linear_wc|mpg_kernel"),
    ("dense EXL3 prompt GEMM", r"_gemm_fold|^_gemm\b|_gemm$"),
    ("dense EXL3 prompt W decode", r"unpack|fdirect"),
    ("dense EXL3 rot_in / had", r"rot_in|had"),
    ("routed experts", r"expert|prompt_kernel|grouped|route|moe"),
    ("GDN / DeltaNet", r"gdn|delta|chunk_|fused_recurrent|solve|kkt|conv1d|recurrent|wy"),
    ("attention", r"attn|attention|flash|sdpa|fmha|softmax|index_qk|topk|sparse"),
    ("hyper-connection", r"_hc|hyper|sinkhorn|mix"),
    ("n-gram / PLE", r"ple|ngram|stage"),
    ("fp16/bf16 matmul (cuBLAS/triton)", r"gemm|gemv|cutlass|sm\d+_|ampere|hopper|matmul|_mm\b|f16_mm"),
    ("norm / elementwise / copy", r"norm|elementwise|vectorized|copy|fill|reduce|cat|index|scatter|gather|silu|act"),
]


def load(path):
    c = sqlite3.connect(path)
    S = dict(c.execute("select id, value from StringIds"))
    K = [(s, e, S.get(dn, "?"), S.get(sn, "?")) for s, e, dn, sn in
         c.execute("select start, end, demangledName, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    return K


def cat(name):
    for label, rx in CATS:
        if re.search(rx, name, re.I):
            return label
    return "other"


def summary(K):
    span = (K[-1][1] - K[0][0]) / 1e6
    busy, cur_s, cur_e = 0, None, None
    for s, e, _, _ in K:                                   # union of kernel intervals (any stream)
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    by_cat, by_k = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for s, e, dn, sn in K:
        by_cat[cat(dn)] += e - s
        key = re.sub(r"<.*", "", sn)[:70]
        by_k[key][0] += e - s
        by_k[key][1] += 1
    return span, busy / 1e6, by_cat, by_k


res = {p: summary(load(p)) for p in sys.argv[1:]}
names = [p.rsplit("/", 1)[-1].replace(".sqlite", "") for p in res]
print("| | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
print("| captured span ms | " + " | ".join(f"{r[0]:.1f}" for r in res.values()) + " |")
print("| GPU busy ms (union) | " + " | ".join(f"{r[1]:.1f} ({100 * r[1] / r[0]:.0f} %)" for r in res.values()) + " |")
cats = sorted({c for r in res.values() for c in r[2]}, key=lambda c: -max(r[2][c] for r in res.values()))
for c in cats:
    print(f"| {c} | " + " | ".join(f"{r[2][c] / 1e6:.1f} ({100 * r[2][c] / 1e6 / r[1]:.1f} %)" for r in res.values()) + " |")
for p, r in res.items():
    print(f"\n== {p.rsplit('/', 1)[-1]}: top kernels (ms, calls, category)")
    for k, (t, n) in sorted(r[3].items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"  {t / 1e6:9.2f} ms x{n:6d}  {k}  [{cat(k)}]")
