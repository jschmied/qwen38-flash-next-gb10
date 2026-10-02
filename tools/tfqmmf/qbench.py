#!/usr/bin/env python3
"""Block-FP8 (Fp8BlockLinear) prompt GEMMs at Flash Next's dense shapes: TFLOPS and an output hash per (shape, rows).
argv: <src> <label>; the tile comes from TF_QMMF_PROMPT_TILE / TF_QMMF_CFG at import."""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear                         # noqa: E402

torch.manual_seed(0)
dev = "cuda"
SHAPES = {"gdn.in_proj_qkv": (2560, 10240), "gdn.in_proj_z": (2560, 6144), "gdn.out_proj": (6144, 2560),
          "attn.q_proj": (2560, 12288), "attn.o_proj": (6144, 2560), "attn.k_proj": (2560, 512)}
out = {"label": sys.argv[2]}
for name, (K, N) in SHAPES.items():
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.float8_e4m3fn)
    s = torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005
    lin = Fp8BlockLinear.from_checkpoint(w, s)
    for M in (512, 2048):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        y = lin.prefill(x)
        h = hashlib.sha256(y.float().cpu().numpy().tobytes()).hexdigest()[:12]
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        best = 1e9
        for _ in range(3):
            a.record()
            for _ in range(10):
                lin.prefill(x)
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) / 10)
        out[f"{name}@{M}"] = {"ms": round(best, 4), "TFLOPS": round(2 * M * N * K / best / 1e9, 1), "hash": h}
print(json.dumps(out), flush=True)
