"""Producer->consumer micro-chain: marginal cost of a mixer-down read after different producer kernels."""
import json, math, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk_venv import bf16sk
N, K, M, CALLS = 336, 10240, 4, 48
ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(CALLS)]
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
small = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
st3 = [torch.randn(3 << 18, device="cuda") for _ in range(CALLS)]           # 3 MiB fp32 each
st12 = [torch.randn(3 << 20, device="cuda") for _ in range(CALLS)]          # 12 MiB fp32 each
src12 = [torch.randn(3 << 20, device="cuda") for _ in range(8)]
acc = torch.zeros(1, device="cuda")
PROD = {"P0_none": None,
        "P1_tiny": lambda i: small.add_(1e-3),
        "P2_write3MiB": lambda i: st3[i].mul_(0.999),
        "P3_write12MiB": lambda i: st12[i].mul_(0.999),
        "P4_read12MiB": lambda i: torch.sum(src12[i % 8], dim=0, out=acc[0])}
CONS = {"triton": lambda i: bf16sk(x, ws[i], 16, 256, 2), "cublas": lambda i: F.linear(x, ws[i])}


def gtime(fn):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream()); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(CALLS): fn(i)
    torch.cuda.synchronize(); v = []
    for _ in range(21):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000 / CALLS)
    v.sort(); return v[len(v) // 2]


res = {}
for pn, p in PROD.items():
    pt = gtime(p) if p else 0.0
    for cn, c in CONS.items():
        both = gtime((lambda i, p=p, c=c: (p(i), c(i))) if p else c)
        res[f"{pn}/{cn}"] = {"producer_us": round(pt, 1), "pair_us": round(both, 1), "consumer_marginal_us": round(both - pt, 1)}
        print(pn, cn, res[f"{pn}/{cn}"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
