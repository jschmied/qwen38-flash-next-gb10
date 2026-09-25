"""Is mixer down slower right after the FP8 blockwise out_proj? Standalone, the in-model call path."""
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
from fn_bf16sk_venv import bf16sk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
M, CALLS = 4, 16
dev = "cuda"
mds = [torch.randn(336, 10240, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(CALLS)]
x_md = torch.randn(M, 10240, device=dev, dtype=torch.bfloat16)
wq = [(torch.randn(2560, 6144, device=dev) * 0.05).to(torch.float8_e4m3fn) for _ in range(CALLS)]
ws = [torch.rand(20, 48, device=dev) * 0.01 + 1e-3 for _ in range(CALLS)]
x_op = torch.randn(M, 6144, device=dev, dtype=torch.bfloat16)
A, As = per_token_group_quant_fp8(x_op, 128, column_major_scales=True)
bf_same = [torch.randn(1280, 6144, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(8)]  # 15.7 MB bf16
tiny = torch.randn(M, 2560, device=dev, dtype=torch.bfloat16)
print("As", tuple(As.shape), As.stride(), flush=True)

def op(i): return ops.cutlass_scaled_mm(A, wq[i].T, out_dtype=torch.bfloat16, scale_a=As, scale_b=ws[i].T)
def md_t(i): return bf16sk(x_md, mds[i], 16, 256, 2)
def md_c(i): return F.linear(x_md, mds[i])
def bfread(i): return bf16sk(x_op, bf_same[i % 8], 64, 256, 1)
def tk(i): return tiny.mul_(1.0001)

def gtime(fn):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream()); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(CALLS): fn(i)
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); v = []
    for _ in range(21):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000 / CALLS)
    v.sort(); return round(v[len(v) // 2], 1)

res = {}
base = {"out_proj": gtime(op), "out_proj+tiny": gtime(lambda i: (op(i), tk(i))), "bf16_same_bytes": gtime(bfread),
        "tiny": gtime(tk)}
res["producers_alone"] = base; print(base, flush=True)
for cn, c in (("triton", md_t), ("cublas", md_c)):
    r = {"C0_alone": gtime(c)}
    r["C1_after_out_proj"] = round(gtime(lambda i: (op(i), c(i))) - base["out_proj"], 1)
    r["C2_after_out_proj+tiny"] = round(gtime(lambda i: (op(i), tk(i), c(i))) - base["out_proj+tiny"], 1)
    r["C3_after_bf16_same_bytes"] = round(gtime(lambda i: (bfread(i), c(i))) - base["bf16_same_bytes"], 1)
    res[cn] = r; print(cn, r, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
