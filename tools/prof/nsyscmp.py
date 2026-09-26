"""Compare two warm nsys node traces (base vs RecoverSSM), per verify step. argv: <base.sqlite> <rssm.sqlite>

Window: the profiled ~150-token request, trimmed 10 % at each end. Steps are counted by the GDN spec kernel
(native `fused_sigmoid_gating_delta_rule_update`, or `_gdn_recoverssm_verify_kernel`) divided by 36 GDN layers.
Per step: kernel time by class, the GPU busy union over all streams, idle, and the medians of the two out_proj
slow-spot kernels that §4o tied to the snapshot write-back.
"""
import collections, re, sqlite3, statistics as st, sys

GDN_LAYERS = 36
CLASSES = [
    ("gdn_spec", r"fused_sigmoid_gating_delta_rule_update|_gdn_recoverssm_verify_kernel"),
    ("rssm_commit", r"_commit_gdn_state_kernel|_prepare_commit_plan_kernel|_compact_conv_state_kernel|ple_recoverssm"),
    ("gdn_other", r"_causal_conv1d|fused_gdn|l2norm_fwd|chunk_gated_delta|_ple_short_conv"),
    ("fp8_gemm", r"fp8_blockwise|per_token_group_quant_8bit"),
    ("nvfp4_moe", r"e2m1|nvfp4|fp16_to_fp4|Moe|moe::|Expert|expandInputRows|doActivation|computeStrides"),
    ("attention", r"_qsa_|flash|fmha|reshape_and_cache"),
    ("ple_gather", r"gather_mapped|ngram|engram"),
    ("bf16_gemm", r"gemvx|wmma_tensorop_bf16|splitKreduce|cublas|nvjet"),
    ("hyper_conn", r"_hc_"),
    ("norm_elementwise", r"norm|elementwise|act_and_mul|silu|Fill|copy|index_"),
]


def load(path):
    c = sqlite3.connect(path)
    S = dict(c.execute("select id, value from StringIds"))
    K = [(s, e, S.get(sn, "?"), S.get(dn, "?"), gx, gy, stream)
         for s, e, sn, dn, gx, gy, stream in
         c.execute("select start,end,shortName,demangledName,gridX,gridY,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    spec = [k for k in K if re.search(CLASSES[0][1], k[2])]
    lo, hi = spec[len(spec) // 10][0], spec[-len(spec) // 10][0]
    W = [k for k in K if lo <= k[0] < hi]
    steps = sum(1 for k in spec if lo <= k[0] < hi) / GDN_LAYERS
    return W, steps, (hi - lo) / 1000


def classify(name):
    for cls, pat in CLASSES:
        if re.search(pat, name, re.I):
            return cls
    return "other"


def summarize(path):
    W, steps, span_us = load(path)
    by = collections.Counter()
    for s, e, sn, dn, gx, gy, stream in W:
        by[classify(sn + " " + dn)] += (e - s) / 1000
    busy, end = 0.0, None
    for s, e, *_ in W:                                    # union of kernel intervals, all streams
        if end is None or s > end:
            busy += (e - s); end = e
        elif e > end:
            busy += (e - end); end = e
    busy /= 1000
    out = {}
    for i, k in enumerate(W):                             # §4o slow spot: FP8 blockwise GEMM grid (1, 20)
        if "fp8_blockwise" in k[3] and k[4] == 1 and k[5] == 20:
            prev = " ".join(W[i - q][2] for q in range(1, 14) if i - q >= 0)
            key = "gdn_out_proj" if re.search(CLASSES[0][1], prev) else "qsa_o_proj"
            out.setdefault(key, []).append((k[1] - k[0]) / 1000)
    names = collections.Counter()
    for s, e, sn, *_ in W:
        names[sn] += (e - s) / 1000
    return {"steps": steps, "ms_per_step": span_us / steps / 1000, "busy_ms": busy / steps / 1000,
            "idle_ms": (span_us - busy) / steps / 1000,
            "class_ms": {k: v / steps / 1000 for k, v in by.items()},
            "out_proj_us": {k: (round(st.median(v), 1), len(v)) for k, v in out.items()},
            "top_kernels_us": [(n, round(v / steps, 1)) for n, v in names.most_common(25)]}


if __name__ == "__main__":
    import json
    a, b = summarize(sys.argv[1]), summarize(sys.argv[2])
    print(json.dumps({"base": a, "rssm": b}, indent=1))
    print(f"\nper step         base     rssm     delta")
    for key in ("ms_per_step", "busy_ms", "idle_ms"):
        print(f"{key:15s} {a[key]:8.3f} {b[key]:8.3f} {b[key]-a[key]:+8.3f}")
    for cls in sorted(set(a["class_ms"]) | set(b["class_ms"]), key=lambda k: -a["class_ms"].get(k, 0)):
        x, y = a["class_ms"].get(cls, 0), b["class_ms"].get(cls, 0)
        print(f"{cls:15s} {x:8.3f} {y:8.3f} {y-x:+8.3f}")
    print("out_proj medians (us, n):", "base", a["out_proj_us"], "| rssm", b["out_proj_us"])
