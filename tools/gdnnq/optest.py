"""Installed FNGDNNQ op vs today's norm + quant, with a non-contiguous z (a slice, as in the model). Prints ONE json."""
import json, torch
import vllm.model_executor.layers.mamba.gdn.fn_gdn_norm_quant as m
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
torch.manual_seed(1); dev = "cuda"; out = {}
for T in (3456, 595, 6):
    x = torch.randn(T, 48, 128, device=dev).bfloat16() * 3
    zbig = torch.randn(T, 48, 256, device=dev).bfloat16(); z = zbig[..., :128]
    w = (1.0 + 0.1 * torch.randn(128, device=dev)).float()
    y = rmsnorm_fn(x, w, None, z=z.contiguous(), eps=1e-6, group_size=None, norm_before_gate=True, activation="silu")
    qa, sa = per_token_group_quant_fp8(y.flatten(-2), 128, column_major_scales=True, use_ue8m0=False)
    qb, sb = torch.ops.vllm.qwen4_exp_gdn_norm_quant(x, z, w, 1e-6, "silu")
    out[T] = {"z_contig": z.is_contiguous(), "A_mismatch": int((qa.view(torch.uint8) != qb.view(torch.uint8)).sum()),
              "of": qa.numel(), "scale_mismatch": int((sa != sb.t()).sum())}
print(json.dumps(out))
