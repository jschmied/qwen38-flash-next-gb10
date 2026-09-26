# SPDX-License-Identifier: Apache-2.0
"""NVFP4 weights for the MTP drafter's BF16 dense linears (jschmied 2026-09-26, local experiment A2, not upstream).

FN_MTP_DENSE_NVFP4=1: after loading, fc_embedding, fc_hidden, self_attn.qkv_proj and self_attn.o_proj of the MTP
drafter keep an NVFP4 copy (fn_nvfp4_head layout: E2M1 + E4M3 per-16 scales + one FP32 global scale) and run through
the draft head's `_nvfp4_rows_gemv_kernel` as a registered custom op. Only DRAFT tokens change; the target verifies
every token, so the served text can change only through acceptance (which, under RecoverSSM, moves late text, §5b).
"""
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)

TARGET_SUFFIXES = ("fc_embedding", "fc_hidden", "self_attn.qkv_proj", "self_attn.o_proj")
_REGISTERED = False
_CALLED = False
# FN_MTP_DENSE_FP8=1: FP8 E4M3 weights with one FP32 scale per output row instead of NVFP4 (half the bytes of BF16)
_FP8 = os.environ.get("FN_MTP_DENSE_FP8", "") == "1"
_LAYERS = tuple(x for x in os.environ.get("FN_MTP_DENSE_LAYERS", "").split(",") if x) or None


@triton.jit
def _fp8_rows_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, sxm, swn, som,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :], mask=nmask[:, None], other=0.0)
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :], mask=mmask[:, None], other=0.0)
        acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    sc = tl.load(s_ptr + offs_n, mask=nmask, other=0.0)
    acc = acc * sc[None, :]
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :], acc, mask=mmask[:, None] & nmask[None, :])


def quantize_fp8_rows(w: torch.Tensor):
    """w float32 [N, K] -> (fp8 e4m3 [N, K], float32 row scales [N])."""
    s = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    return (w / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous(), s.contiguous()


def fp8_rows_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, block_n: int = 64, block_k: int = 128,
                  num_warps: int = 4) -> torch.Tensor:
    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, 16))
    _fp8_rows_gemv_kernel[grid](x, w, s, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                                BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps)
    return out


def _impl_fp8(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    global _CALLED
    if not _CALLED:
        _CALLED = True
        logger.warning("FNMTPDENSE8 path taken: FP8 rows GEMV, x %s -> %d", tuple(x.shape), w.shape[0])
    x2 = x.reshape(-1, x.shape[-1])
    out = fp8_rows_gemv(x2.to(torch.bfloat16), w, s)
    return out.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])


def _fake_fp8(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def _impl(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, g: float) -> torch.Tensor:
    from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import nvfp4_rows_gemv
    global _CALLED
    if not _CALLED:
        _CALLED = True
        logger.warning("FNMTPDENSE4 path taken: NVFP4 rows GEMV, x %s -> %d", tuple(x.shape), q.shape[0])
    x2 = x.reshape(-1, x.shape[-1])
    out = nvfp4_rows_gemv(x2.to(torch.bfloat16), q, s, g)
    return out.to(x.dtype).reshape(*x.shape[:-1], q.shape[0])


def _fake(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, g: float) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], q.shape[0]))


def _register() -> None:
    global _REGISTERED
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name="fn_nvfp4_linear", op_func=_impl, fake_impl=_fake)
        direct_register_custom_op(op_name="fn_fp8_rows_linear", op_func=_impl_fp8, fake_impl=_fake_fp8)
        _REGISTERED = True


class FnNvfp4DenseMethod(UnquantizedLinearMethod):
    """Loads like the stock BF16 method, then quantizes once; apply() runs the NVFP4 rows GEMV."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import quantize_nvfp4_rows
        w = layer.weight.data
        if _FP8:
            q8, s8 = quantize_fp8_rows(w.float())
            layer.register_buffer("_fn8_w", q8, persistent=False)
            layer.register_buffer("_fn8_s", s8, persistent=False)
            logger.warning("FNMTPDENSE8: %s %s -> FP8 rows (%.1f -> %.1f MiB)", getattr(layer, "prefix", "?"),
                           tuple(w.shape), w.numel() * w.element_size() / 2**20, q8.numel() / 2**20)
            return
        q, s, g = quantize_nvfp4_rows(w.float())
        layer.register_buffer("_fn4_q", q, persistent=False)
        layer.register_buffer("_fn4_s", s, persistent=False)
        layer._fn4_g = float(g)
        logger.warning("FNMTPDENSE4: %s %s -> NVFP4 (%.1f -> %.1f MiB)", getattr(layer, "prefix", "?"),
                       tuple(w.shape), w.numel() * w.element_size() / 2**20,
                       (q.numel() + s.numel()) / 2**20)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if _FP8:
            out = torch.ops.vllm.fn_fp8_rows_linear(x, layer._fn8_w, layer._fn8_s)
        else:
            out = torch.ops.vllm.fn_nvfp4_linear(x, layer._fn4_q, layer._fn4_s, layer._fn4_g)
        return out if bias is None else out + bias


def swap_methods(model: torch.nn.Module) -> int:
    """Before process_weights_after_loading: give the target BF16 linears the NVFP4 method."""
    _register()
    n = 0
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if (name.endswith(_LAYERS or TARGET_SUFFIXES) and type(qm) is UnquantizedLinearMethod
                and getattr(mod, "weight", None) is not None and mod.weight.dtype == torch.bfloat16):
            mod.quant_method = FnNvfp4DenseMethod()
            mod.prefix = name
            n += 1
    logger.warning("FNMTPDENSE%s armed: %d MTP dense linears to %s", "8" if _FP8 else "4", n,
                   "FP8 rows" if _FP8 else "NVFP4")
    return n
