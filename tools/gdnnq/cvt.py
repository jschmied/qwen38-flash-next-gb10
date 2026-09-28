"""Which fp32->bf16 conversion inside a kernel reproduces FLA's stored y? Variants vs rmsnorm_fn over 21M elements."""
import json, torch, triton, triton.language as tl
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
@triton.jit
def _y(X, Z, W, Y, M, eps, N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, MODE: tl.constexpr):
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
    if MODE == 0:
        tl.store(Y + row_offsets + col_offsets, y, mask=mask)                      # implicit (FLA)
    elif MODE == 1:
        tl.store(Y + row_offsets + col_offsets, y.to(tl.bfloat16), mask=mask)      # explicit .to
    elif MODE == 2:
        tl.store(Y + row_offsets + col_offsets, y.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    else:
        b = y.to(tl.uint32, bitcast=True)
        r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        tl.store(Y + row_offsets + col_offsets, r.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)
torch.manual_seed(0); dev = "cuda"; NH, D = 48, 128; out = {}
w = (1.0 + 0.1 * torch.randn(D, device=dev)).float()
x = torch.randn(3456, NH, D, device=dev).bfloat16() * 3; z = torch.randn(3456, NH, D, device=dev).bfloat16()
ya = rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu").reshape(-1, D)
M = 3456 * NH
for mode in range(4):
    yb = torch.empty(M, D, device=dev, dtype=torch.bfloat16)
    _y[(triton.cdiv(M, 4),)](x, z, w, yb, M, 1e-6, N=D, BLOCK_N=128, ROWS_PER_BLOCK=4, MODE=mode, num_warps=1)
    out[["implicit", "to", "to_rtne", "manual_rne"][mode]] = int((ya != yb).sum())
print(json.dumps(out))
