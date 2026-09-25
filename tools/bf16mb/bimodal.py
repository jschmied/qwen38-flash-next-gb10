"""Repeat the same mixer-down chain 40x (Triton and cuBLAS alternating), 0.5 s apart; print each median."""
import sys, time, json, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk_venv import bf16sk
N, K, M, CALLS = 336, 10240, 4, 48
ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(CALLS)]
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
def build(fn):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream()); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(CALLS): fn(i)
    torch.cuda.synchronize(); return g
def t(g, reps=21):
    v = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000 / CALLS)
    v.sort(); return round(v[len(v) // 2], 1), round(v[0], 1), round(v[-1], 1)
gt = build(lambda i: bf16sk(x, ws[i], 16, 256, 2)); gc = build(lambda i: F.linear(x, ws[i]))
out = []
for r in range(40):
    a = t(gt); b = t(gc); out.append((r, a, b)); print(r, "triton med/min/max", a, "cublas", b, flush=True); time.sleep(0.5)
# sustained: 2 s of back-to-back replays, then measure
for r in range(5):
    t0 = time.time()
    while time.time() - t0 < 2: gt.replay()
    torch.cuda.synchronize(); print("after 2s load", r, "triton", t(gt), "cublas", t(gc), flush=True)
