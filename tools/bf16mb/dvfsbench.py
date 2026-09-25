import json, sys, time, torch
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
heavy = [torch.randn(32 << 20, device=dev) for _ in range(4)]         # 4 x 128 MiB (rotated: never L2-resident)
tiny = torch.randn(4096, device=dev); acc = torch.zeros(1, device=dev)
# calibrate the spin kernel to ~300 us
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
cyc = 700_000
for _ in range(3):
    e0.record(); torch.cuda._sleep(cyc); e1.record(); e1.synchronize(); us = e0.elapsed_time(e1) * 1000; cyc = int(cyc * 300 / us)
e0.record(); acc.add_(heavy[0].sum()); e1.record(); e1.synchronize(); heavy_us = e0.elapsed_time(e1) * 1000
PRODUCERS = {
    "idle_gap": None,
    "spin_300us_no_dram": lambda i: torch.cuda._sleep(cyc),
    "tiny_x30_l2": lambda i: [tiny.mul_(1.0001) for _ in range(30)],
    "dram_heavy_read": lambda i: acc.add_(heavy[i % 4].sum()),
}
res = {"spin_cycles": cyc, "heavy_read_us": round(heavy_us, 1)}
for name, prod in PRODUCERS.items():
    t_op, t_md, t_pr = [], [], []
    for rep in range(30):
        for i in range(CALLS):
            e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            e[0].record()
            if prod: prod(i)
            e[1].record(); ops.cutlass_scaled_mm(A, wq[i].T, out_dtype=torch.bfloat16, scale_a=As, scale_b=ws[i].T)
            e[2].record(); bf16sk(x_md, mds[i], 16, 256, 2); e[3].record(); e[3].synchronize()
            if rep >= 3:
                t_pr.append(e[0].elapsed_time(e[1]) * 1000); t_op.append(e[1].elapsed_time(e[2]) * 1000); t_md.append(e[2].elapsed_time(e[3]) * 1000)
    m = lambda v: round(sorted(v)[len(v) // 2], 1)
    res[name] = {"producer_us": m(t_pr), "out_proj_us": m(t_op), "mixer_down_us": m(t_md)}
    print(f"{name:22s} producer {m(t_pr):7.1f} us | out_proj {m(t_op):6.1f} us | mixer_down {m(t_md):6.1f} us", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
