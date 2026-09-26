# SPDX-License-Identifier: Apache-2.0
"""NVFP4 weights for the MTP drafter's BF16 dense linears (jschmied 2026-09-26, local experiment A2, not upstream).

FN_MTP_DENSE_NVFP4=1: after loading, fc_embedding, fc_hidden, self_attn.qkv_proj and self_attn.o_proj of the MTP
drafter keep an NVFP4 copy (fn_nvfp4_head layout: E2M1 + E4M3 per-16 scales + one FP32 global scale) and run through
the draft head's `_nvfp4_rows_gemv_kernel` as a registered custom op. Only DRAFT tokens change; the target verifies
every token, so the served text can change only through acceptance (which, under RecoverSSM, moves late text, §5b).
"""
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)

TARGET_SUFFIXES = ("fc_embedding", "fc_hidden", "self_attn.qkv_proj", "self_attn.o_proj")
_REGISTERED = False
_CALLED = False


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
        _REGISTERED = True


class FnNvfp4DenseMethod(UnquantizedLinearMethod):
    """Loads like the stock BF16 method, then quantizes once; apply() runs the NVFP4 rows GEMV."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import quantize_nvfp4_rows
        w = layer.weight.data
        q, s, g = quantize_nvfp4_rows(w.float())
        layer.register_buffer("_fn4_q", q, persistent=False)
        layer.register_buffer("_fn4_s", s, persistent=False)
        layer._fn4_g = float(g)
        logger.warning("FNMTPDENSE4: %s %s -> NVFP4 (%.1f -> %.1f MiB)", getattr(layer, "prefix", "?"),
                       tuple(w.shape), w.numel() * w.element_size() / 2**20,
                       (q.numel() + s.numel()) / 2**20)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.ops.vllm.fn_nvfp4_linear(x, layer._fn4_q, layer._fn4_s, layer._fn4_g)
        return out if bias is None else out + bias


def swap_methods(model: torch.nn.Module) -> int:
    """Before process_weights_after_loading: give the target BF16 linears the NVFP4 method."""
    _register()
    n = 0
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if (name.endswith(TARGET_SUFFIXES) and type(qm) is UnquantizedLinearMethod
                and getattr(mod, "weight", None) is not None and mod.weight.dtype == torch.bfloat16):
            mod.quant_method = FnNvfp4DenseMethod()
            mod.prefix = name
            n += 1
    logger.warning("FNMTPDENSE4 armed: %d MTP dense linears to NVFP4", n)
    return n
