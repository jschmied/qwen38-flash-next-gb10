# #55122 kernel bench: wheel _C vs merge-base build vs PR-head build vs exact torch.topk, on Qwen3.8-Flash-Next
# QSA shapes (block_topk = indexer_budget 2048 / compress_ratio 4 = 512).
#   argv: <_C_detbase.so> <_C_detpr.so>
# Correctness pre-checks gate the timing: a failing check prints VOID and exits 2.
import sys, time, statistics, itertools, json
import torch
torch.ops.load_library(sys.argv[1]); torch.ops.load_library(sys.argv[2])
import vllm._custom_ops  # noqa: F401  (loads _C)

dev = "cuda"; K = 512
WS = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)   # qsa_indexer._TOPK_WORKSPACE_BYTES


def kern(op):
    def f(logits, lengths, out):
        op(logits, lengths, out, WS, K, logits.shape[1])
    return f


def exact(logits, lengths, out, sort=False):
    n = logits.shape[1]; k = min(K, n)
    col = torch.arange(n, device=logits.device)
    masked = logits.masked_fill(col[None, :] >= lengths.to(torch.int64)[:, None], float("-inf"))
    vals, idx = torch.topk(masked, k, dim=1)
    idx = idx.masked_fill(torch.isinf(vals), -1)
    if sort:
        key = idx.masked_fill(idx < 0, 2**31 - 1).sort(dim=1).values
        idx = key.masked_fill(key == 2**31 - 1, -1)
    if k < K:
        out.fill_(-1)
    out[:, :k].copy_(idx.to(torch.int32))


ARMS = {
    "wheel": kern(torch.ops._C.persistent_topk),
    "base": kern(torch.ops._C_detbase.persistent_topk),
    "pr": kern(torch.ops._C_detpr.persistent_topk),
    "exact": exact,
    "exact_sorted": lambda l, n, o: exact(l, n, o, sort=True),
}


def make(rows, n, kind, lengths_kind, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(rows, n, device=dev, generator=g)
    if kind == "ties":            # FP8-like coarse grid: exact ties at the k-th value are certain
        x = torch.round(x * 8) / 8
    if lengths_kind == "full":
        L = torch.full((rows,), n, dtype=torch.int32, device=dev)
    else:                         # causal prefill chunk: last `rows` tokens of a context of 4*n tokens
        T = 4 * n
        pos = torch.arange(T - rows, T, device=dev)
        L = (pos // 4 + 1).clamp(max=n).to(torch.int32)
    return x, L


def sets(out):
    return [sorted(v for v in r if v >= 0) for r in out.tolist()]


def check():
    bad = []
    for rows, n, lk in ((4, 2048, "full"), (64, 8192, "full"), (4096, 2048, "causal"), (4096, 1024, "causal")):
        x, L = make(rows, n, "rand", lk, seed=1)
        ref = torch.empty(rows, K, dtype=torch.int32, device=dev); exact(x, L, ref)
        rs = sets(ref)
        for a in ("wheel", "base", "pr"):
            o = torch.full((rows, K), -7, dtype=torch.int32, device=dev); ARMS[a](x, L, o)
            if sets(o) != rs:
                bad.append(f"set mismatch {a} rows={rows} n={n} {lk}")
    for rows, n, lk in ((4, 8192, "full"), (64, 8192, "full"), (4096, 2048, "causal")):
        x, L = make(rows, n, "ties", lk, seed=2)
        first = None; distinct = 1
        for i in range(20):
            o = torch.empty(rows, K, dtype=torch.int32, device=dev); ARMS["pr"](x, L, o)
            if first is None: first = o.clone()
            elif not torch.equal(o, first): distinct += 1
        if distinct != 1:
            bad.append(f"pr not bitwise repeatable on ties rows={rows} n={n} ({distinct}/20 differ)")
        # validity: every selected value >= the k-th largest visible value
        ref = torch.empty(rows, K, dtype=torch.int32, device=dev); exact(x, L, ref)
        for r in range(0, rows, max(1, rows // 16)):
            sel = [v for v in first[r].tolist() if v >= 0]; rv = [v for v in ref[r].tolist() if v >= 0]
            if len(sel) != len(rv) or (sel and x[r, sel].min() < x[r, rv].min()):
                bad.append(f"pr invalid top-k on ties rows={rows} n={n} row={r}"); break
    return bad


def time_arm(f, x, L, out, iters=50, batches=5):
    for _ in range(10): f(x, L, out)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = []
    for _ in range(batches):
        torch.cuda.synchronize(); s.record()
        for _ in range(iters): f(x, L, out)
        e.record(); e.synchronize(); res.append(s.elapsed_time(e) / iters * 1000)
    return statistics.median(res)


def main():
    bad = check()
    print(f"prechecks: {'OK' if not bad else 'FAIL'}", flush=True)
    for b in bad: print("  " + b, flush=True)
    if bad:
        print("== VOID =="); sys.exit(2)
    cases = [("decode", r, n, "rand", "full") for r in (4, 8, 16, 32, 64) for n in (2048, 4096, 8192)]
    cases += [("decode", r, 8192, "ties", "full") for r in (4, 16, 64)]
    cases += [("prefill", 4096, n, d, "causal") for n in (1024, 2048, 4096, 8192) for d in ("rand", "ties")]
    names = list(ARMS)
    print(f"{'phase':7} {'rows':>4} {'n':>5} {'data':5} | " + " ".join(f"{a:>12}" for a in names)
          + " | pr/base  pr/wheel  exact/pr", flush=True)
    rows_out = []
    for ph, rows, n, d, lk in cases:
        x, L = make(rows, n, d, lk)
        out = torch.empty(rows, K, dtype=torch.int32, device=dev)
        t = {a: [] for a in names}
        for rnd in range(2):                     # two rounds, arm order reversed in the second
            order = names if rnd == 0 else names[::-1]
            for a in order:
                t[a].append(time_arm(ARMS[a], x, L, out))
        med = {a: statistics.median(v) for a, v in t.items()}
        rng = {a: (min(v), max(v)) for a, v in t.items()}
        rec = dict(phase=ph, rows=rows, n=n, data=d, us={a: [round(v, 2) for v in t[a]] for a in names},
                   pr_base=med["pr"] / med["base"], pr_wheel=med["pr"] / med["wheel"], exact_pr=med["exact"] / med["pr"],
                   base_wheel=med["base"] / med["wheel"])
        rows_out.append(rec)
        print(f"{ph:7} {rows:4d} {n:5d} {d:5} | " + " ".join(f"{rng[a][0]:6.1f}-{rng[a][1]:<5.1f}" for a in names)
              + f" | {rec['pr_base']:6.2f}  {rec['pr_wheel']:7.2f}  {rec['exact_pr']:7.2f}", flush=True)
    json.dump(rows_out, open(sys.argv[3] if len(sys.argv) > 3 else "/opt/llm/runners/results/topk55122-kbench.json", "w"), indent=1)
    print("== KBENCH DONE ==", flush=True)


main()
