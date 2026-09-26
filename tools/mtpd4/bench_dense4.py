"""A2 microbench: MTP dense shapes, BF16 F.linear vs the NVFP4 rows GEMV (L2 flushed, CUDA-graph replay, M=1 and 4).
Also the accuracy of the NVFP4 op vs its own dequantized weight (kernel check) and vs BF16 (quantization error)."""
import torch, torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import quantize_nvfp4_rows, dequantize_nvfp4_rows, nvfp4_rows_gemv
torch.manual_seed(0); dev = "cuda"
SHAPES = {"qkv": (13312, 2560), "o_proj": (2560, 6144), "fc_embedding": (2560, 2560), "fc_hidden": (2560, 2560)}
flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
def timeit(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); g = torch.cuda.CUDAGraph(); gz = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): flush.zero_(); fn()
        with torch.cuda.graph(gz, stream=s): flush.zero_()
    torch.cuda.synchronize()
    def t(gr):
        v = []
        for _ in range(50):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); gr.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000)
        v.sort(); return v[len(v) // 2]
    return t(g) - t(gz)
tot = {1: [0, 0], 4: [0, 0]}
for name, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    q, s, g = quantize_nvfp4_rows(w.float()); wd = dequantize_nvfp4_rows(q, s, g)
    for M in (1, 4):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = F.linear(x.float(), wd); got = nvfp4_rows_gemv(x, q, s, g)
        kerr = ((got - ref).norm() / ref.norm()).item()
        qerr = ((got - F.linear(x.float(), w.float())).norm() / ref.norm()).item()
        tb = timeit(lambda: F.linear(x, w)); tq = timeit(lambda: nvfp4_rows_gemv(x, q, s, g))
        tot[M][0] += tb; tot[M][1] += tq
        print(f"{name:13s} M={M} bf16 {tb:7.1f} us  nvfp4 {tq:7.1f} us  ({tq/tb:.2f}x)  kernel-rel-err {kerr:.1e}  quant-rel-err {qerr:.3f}", flush=True)
for M in (1, 4):
    print(f"TOTAL M={M}: bf16 {tot[M][0]:.0f} us -> nvfp4 {tot[M][1]:.0f} us, saves {tot[M][0]-tot[M][1]:.0f} us per draft step")

# FP8 per-row variant
from importlib import util
spec = util.spec_from_file_location("fnd", __file__.replace("bench_dense4.py", "fn_nvfp4_dense.py"))
fnd = util.module_from_spec(spec); spec.loader.exec_module(fnd)
tot8 = {1: [0, 0], 4: [0, 0]}
for name, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    w8, s8 = fnd.quantize_fp8_rows(w.float())
    for M in (1, 4):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = F.linear(x.float(), w8.float() * s8[:, None]); got = fnd.fp8_rows_gemv(x, w8, s8)
        kerr = ((got - ref).norm() / ref.norm()).item(); qerr = ((got - F.linear(x.float(), w.float())).norm() / ref.norm()).item()
        tb = timeit(lambda: F.linear(x, w)); tq = timeit(lambda: fnd.fp8_rows_gemv(x, w8, s8))
        tot8[M][0] += tb; tot8[M][1] += tq
        print(f"FP8 {name:13s} M={M} bf16 {tb:7.1f} us  fp8 {tq:7.1f} us  ({tq/tb:.2f}x)  kernel-rel-err {kerr:.1e}  quant-rel-err {qerr:.3f}", flush=True)
for M in (1, 4):
    print(f"FP8 TOTAL M={M}: bf16 {tot8[M][0]:.0f} us -> fp8 {tot8[M][1]:.0f} us, saves {tot8[M][0]-tot8[M][1]:.0f} us per draft step")
