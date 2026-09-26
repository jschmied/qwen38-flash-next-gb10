"""GDN projection shapes at decode: NVFP4 W4A16 Marlin vs BF16 cuBLAS, L2 flushed, graph replay, M = 4 (c=1 verify)
and 16 (c=4). Byte floors at 220 GB/s: FP8 (current) = N*K + block scales; NVFP4 = N*K/2 + N*K/16."""
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear, rand_marlin_weight_nvfp4_like)
dev = "cuda"; torch.manual_seed(0)
SHAPES = {"in_proj_qkvz": (16384, 2560), "out_proj": (2560, 6144)}
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
        for _ in range(60):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); gr.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000)
        v.sort(); return v[len(v) // 2]
    return t(g) - t(gz)
ws = marlin_make_workspace_new(torch.device(dev))
for name, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    wref, qw, sc, gs = rand_marlin_weight_nvfp4_like(w, 16)
    fp8_floor = (N * K + (N // 128) * (K // 128) * 4) / 220e3
    fp4_floor = (N * K / 2 + N * K / 16) / 220e3
    for M in (4, 16):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        out = apply_fp4_marlin_linear(x, qw, sc, gs, ws, N, K)
        ref = x.float() @ wref.float()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        tm = timeit(lambda: apply_fp4_marlin_linear(x, qw, sc, gs, ws, N, K))
        tb = timeit(lambda: torch.nn.functional.linear(x, w))
        print(f"{name:13s} M={M:2d} marlin-nvfp4 {tm:6.1f} us (floor {fp4_floor:5.1f})  bf16 {tb:6.1f} us  "
              f"fp8-floor {fp8_floor:5.1f} us  kernel-rel-err {err:.1e}", flush=True)
