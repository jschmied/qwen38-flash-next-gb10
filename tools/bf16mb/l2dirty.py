import json, sys, torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk_venv import bf16sk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
dev = "cuda"; M, CALLS = 4, 8
wq = [(torch.randn(2560, 6144, device=dev) * 0.05).to(torch.float8_e4m3fn) for _ in range(CALLS)]
ws = [torch.rand(20, 48, device=dev) * 0.01 + 1e-3 for _ in range(CALLS)]
x_op = torch.randn(M, 6144, device=dev, dtype=torch.bfloat16); A, As = per_token_group_quant_fp8(x_op, 128, column_major_scales=True)
mds = [torch.randn(336, 10240, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(CALLS)]
x_md = torch.randn(M, 10240, device=dev, dtype=torch.bfloat16)
state = torch.randn(12 << 18, device=dev)                         # 12 MiB fp32 "state"
acc = torch.zeros(1, device=dev)
res = {}
for mib in (0, 3, 6, 12):
    n = mib << 18
    def step(i):
        if n:
            acc.add_(state[:n].sum())          # read: bring the lines into L2 (clean)
            state[:n].mul_(1.0001)             # overwrite in place: fast, dirty lines in L2
        ops.cutlass_scaled_mm(A, wq[i].T, out_dtype=torch.bfloat16, scale_a=As, scale_b=ws[i].T)
        bf16sk(x_md, mds[i], 16, 256, 2)
    for i in range(3): step(i)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream()); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(CALLS): step(i)
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
    d = {"out_proj": [], "mixer": [], "write": []}
    for e in p.events():
        if e.device_type.name != "CUDA": continue
        if "fp8_blockwise" in e.name: d["out_proj"].append(e.device_time)
        elif "_partial" in e.name or "_reduce" in e.name: d["mixer"].append(e.device_time)
        elif "mul" in e.name.lower() or ("elementwise" in e.name and n): d["write"].append(e.device_time)
    med = lambda v: round(sorted(v)[len(v)//2], 1) if v else None
    mix = [a + b for a, b in zip(d["mixer"][0::2], d["mixer"][1::2])] if d["mixer"] else []
    res[mib] = {"out_proj_us": med(d["out_proj"]), "mixer_us(partial+reduce)": med(mix), "write_us": med(d["write"])}
    print(f"dirty {mib:2d} MiB: out_proj {res[mib]['out_proj_us']} us | mixer {res[mib]['mixer_us(partial+reduce)']} us | overwrite kernel {res[mib]['write_us']} us", flush=True)
    del g
json.dump(res, open(sys.argv[1], "w"), indent=1)
