"""Our kernel's bf16 y vs rmsnorm_fn's, element-wise; float64 reference at the differences."""
import json, torch, triton, triton.language as tl
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
@triton.jit
def _y(X, Z, W, Y, M, eps, N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK); cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * N; col_offsets = cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(Z + row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y *= z * tl.sigmoid(z)
    tl.store(Y + row_offsets + col_offsets, y, mask=mask)
torch.manual_seed(0); dev = "cuda"; NH, D = 48, 128; out = {}
for wdt in (torch.float32, torch.bfloat16):
    w = (1.0 + 0.1 * torch.randn(D, device=dev)).to(wdt)
    x = torch.randn(3456, NH, D, device=dev).bfloat16() * 3; z = torch.randn(3456, NH, D, device=dev).bfloat16()
    ya = rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu").reshape(-1, D)
    M = 3456 * NH; yb = torch.empty(M, D, device=dev, dtype=torch.bfloat16)
    _y[(triton.cdiv(M, 4),)](x, z, w, yb, M, 1e-6, N=D, BLOCK_N=128, ROWS_PER_BLOCK=4, num_warps=1)
    d = (ya != yb).nonzero()
    x64 = x.reshape(-1, D).double(); z64 = z.reshape(-1, D).double()
    ref = x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + 1e-6) * w.double() * z64 * torch.sigmoid(z64)
    rows = []
    for r, c in d[:6].tolist():
        rows.append({"a_fla": float(ya[r, c]), "b_ours": float(yb[r, c]), "ref64": float(ref[r, c])})
    out[str(wdt)[6:]] = {"n_diff": int(d.shape[0]), "of": int(ya.numel()), "fla_closer": sum(1 for r in rows if abs(r["a_fla"]-r["ref64"]) < abs(r["b_ours"]-r["ref64"])), "samples": rows}
print(json.dumps(out))
