# SPDX-License-Identifier: Apache-2.0
# FNGDNNQ (2026-09-28, speed-of-light §5af): the GDN output's gated RMSNorm (FLA layer_norm_fwd_kernel, norm_before_gate,
# head dim 128) and out_proj's per-token-group FP8 quant (group 128, column-major scales, CUDA per_token_group_quant)
# in one Triton kernel. Head dim = quant group, so one row = one (token, head) = one group. The norm expressions and
# tile ([4, 128], 1 warp) are FLA's; the quant is the CUDA kernel's formula with exact division. Drift-level vs the two
# kernels (≤ 1 bf16 ulp on ~3e-6 of y elements, see §5af). Env-gated by FN_GDNNQ=1 in qwen_gdn_linear_attn.py.
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _gdn_norm_quant(X, Z, W, Q, S, M, T, NH, sx_t, sx_h, sz_t, sz_h, eps, qeps,
                    N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, ACTIVATION: tl.constexpr):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    tok = rows // NH
    head = rows % NH
    col_offsets = cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + (tok * sx_t + head * sx_h)[:, None] + col_offsets, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(Z + (tok * sz_t + head * sz_h)[:, None] + col_offsets, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    yf = y.to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=1), qeps)
    s = tl.math.div_rn(absmax, tl.full(absmax.shape, 448.0, tl.float32))
    q = tl.math.div_rn(yf, s[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(Q + rows[:, None] * N + col_offsets, q.to(tl.float8e4nv), mask=mask)
    tl.store(S + head * T + tok, s, mask=rows < M)


def _norm_quant(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float,
                activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """x, z: [T, NH, 128] (last dim contiguous). Returns A [T, NH*128] e4m3 and the scales as [NH, T] fp32
    (pass ``.t()`` as the column-major ``As``)."""
    T, NH, D = x.shape
    M = T * NH
    q = torch.empty(T, NH * D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(NH, T, device=x.device, dtype=torch.float32)
    if M:
        logger.info_once("FNGDNNQ fused GDN norm+quant ran (T=%d)", T)
        _gdn_norm_quant[(triton.cdiv(M, 4),)](x, z, weight, q, s, M, T, NH, x.stride(0), x.stride(1), z.stride(0),
                                              z.stride(1), eps, 1e-10, N=D, BLOCK_N=128, ROWS_PER_BLOCK=4,
                                              ACTIVATION=activation, num_warps=1)
    return q, s


def _norm_quant_fake(x, z, weight, eps, activation):
    T, NH, D = x.shape
    return (x.new_empty((T, NH * D), dtype=torch.float8_e4m3fn), x.new_empty((NH, T), dtype=torch.float32))


direct_register_custom_op(op_name="qwen4_exp_gdn_norm_quant", op_func=_norm_quant, fake_impl=_norm_quant_fake)


def gdn_output_fused(attn, core_attn_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Gated RMSNorm + group FP8 quant fused, then out_proj's own CUTLASS block-scaled GEMM on that A/As."""
    q, s = torch.ops.vllm.qwen4_exp_gdn_norm_quant(core_attn_out, z, attn.norm.weight, attn.norm.eps,
                                                   attn.norm.activation)
    k = attn.out_proj.quant_method.kernel
    p = k._get_layer_params(attn.out_proj)
    out = k.apply_block_scaled_mm(A=q, B=p.weight, As=s.t(), Bs=p.block_scale)
    return out.to(k.config.out_dtype)


def gdn_output_fusable(attn) -> bool:
    """Every precondition of the fused path, checked once per layer."""
    try:
        n, op = attn.norm, attn.out_proj
        qm = op.quant_method
        k = getattr(qm, "kernel", None)
        if type(k).__name__ != "CutlassFp8BlockScaledMMKernel" or not getattr(k, "apply_input_quant", False):
            return False
        qf = k.quant_fp8
        if not (qf.column_major_scales and not qf.use_ue8m0 and qf.group_size == 128):
            return False
        if type(getattr(qm, "fmt", None)).__name__ != "FormatScheme":
            return False
        return (n.group_size is None and n.norm_before_gate and getattr(n, "bias", None) is None
                and attn.head_v_dim == 128 and op.bias is None and getattr(op, "tp_size", 1) == 1
                and n.activation in ("silu", "swish", "sigmoid"))
    except AttributeError:
        return False
