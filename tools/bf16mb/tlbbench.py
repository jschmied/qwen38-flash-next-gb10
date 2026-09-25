"""Does a TLB-sweeping producer slow the next weight reads? Standalone, exact in-model kernels."""
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk_venv import bf16sk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
dev = "cuda"; M, CALLS = 4, 8
big = torch.empty(12 << 30, dtype=torch.uint8, device=dev)           # 12 GiB pool to sweep
big.view(torch.int32)[:: (1 << 18)].fill_(1)                         # touch every 1 MiB once so it is mapped
f32 = big.view(torch.float32)
wq = [(torch.randn(2560, 6144, device=dev) * 0.05).to(torch.float8_e4m3fn) for _ in range(CALLS)]
ws = [torch.rand(20, 48, device=dev) * 0.01 + 1e-3 for _ in range(CALLS)]
x_op = torch.randn(M, 6144, device=dev, dtype=torch.bfloat16); A, As = per_token_group_quant_fp8(x_op, 128, column_major_scales=True)
mds = [torch.randn(336, 10240, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(CALLS)]
x_md = torch.randn(M, 10240, device=dev, dtype=torch.bfloat16)
out = torch.zeros(1, device=dev)
def sweep_fn(span_gib, stride):
    idx = torch.arange(0, int(span_gib * (1 << 30)) // 4, stride // 4, device=dev, dtype=torch.int64)
    return (lambda: out.add_(f32[idx].sum())), idx.numel()
def run(producer):
    """graph of CALLS x [producer?, out_proj, mixer_down]; time each consumer with events inside a replay loop."""
    t_op, t_md = [], []
    for rep in range(25):
        for i in range(CALLS):
            if producer: producer()
            e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            e[0].record(); ops.cutlass_scaled_mm(A, wq[i].T, out_dtype=torch.bfloat16, scale_a=As, scale_b=ws[i].T)
            e[1].record(); bf16sk(x_md, mds[i], 16, 256, 2); e[2].record()
            e[2].synchronize()
            if rep >= 3: t_op.append(e[0].elapsed_time(e[1]) * 1000); t_md.append(e[1].elapsed_time(e[2]) * 1000)
    t_op.sort(); t_md.sort()
    return round(t_op[len(t_op) // 2], 1), round(t_md[len(t_md) // 2], 1)
res = {"none": run(None)}
print("no producer: out_proj %.1f us, mixer_down %.1f us" % res["none"], flush=True)
for span, stride in ((0.0625, 64 << 10), (1, 64 << 10), (4, 64 << 10), (12, 64 << 10), (4, 2 << 20), (12, 2 << 20)):
    p, n = sweep_fn(span, stride); r = run(p); res[f"{span}GiB/{stride>>10}KiB"] = r
    print(f"after sweep {span:6.3f} GiB stride {stride>>10:5d} KiB ({n:7d} touches): out_proj {r[0]:6.1f} us, mixer_down {r[1]:6.1f} us", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
