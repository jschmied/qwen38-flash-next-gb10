# SPDX-License-Identifier: Apache-2.0
"""NVFP4 W4A16 target lm_head for Qwen4Exp (jschmied 2026-09-30, local experiment, not upstream). FN_TARGET_HEAD_NVFP4=1.

The target's lm_head is FP8 block-scaled (248,320 x 2,560, 128x128 blocks; 0.59 GiB + scales) and is read once per
verify. This re-quantizes it at the first compute_logits call to NVFP4 (E2M1 values, one E4M3 scale per 16 along K, one
FP32 global scale; 0.30 + 0.04 GiB) and computes the logits with the draft head's Triton W4A16 GEMV
(`fn_nvfp4_head.nvfp4_rows_gemv`). Unlike the draft head this changes the target's distribution: a precision cut that
needs a quality screen. The FP8 weights stay in place (the MTP drafter's vocab slice reads them).

Built from the FP8 dequant (w * block scale), in row chunks so the float32 temporary stays ~0.3 GiB; one global scale
over the whole head (computed first), so every chunk shares the kernel's single `g`."""
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_CHUNK = 32768


def _fp8_rows(head, r0, r1, bn, bk):
    rows = head.weight[r0:r1].to(torch.float32)
    srows = head.weight_scale[r0 // bn:(r1 + bn - 1) // bn].to(torch.float32)
    srows = srows.repeat_interleave(bn, dim=0)[: r1 - r0]
    return rows * srows.repeat_interleave(bk, dim=1)


def _quant(w: torch.Tensor, g: float):
    n, k = w.shape
    grp = w.view(n, k // 16, 16)
    s = (grp.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = s.to(torch.float32) * g
    x = torch.where(sf[..., None] > 0, grp / sf[..., None].clamp(min=1e-30), torch.zeros_like(grp))
    code = torch.bucketize(x.abs().clamp(max=6.0), torch.tensor(_MID, device=w.device, dtype=torch.float32))
    nib = (code | ((x < 0).to(code.dtype) << 3)).to(torch.uint8).view(n, k)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous(), s.contiguous()


def build(model):
    head = model.lm_head
    w = getattr(head, "weight", None)
    s = getattr(head, "weight_scale", None)
    if w is None or s is None or w.dim() != 2 or s.dim() != 2 or w.dtype != torch.float8_e4m3fn \
            or getattr(head, "tp_size", 1) != 1:
        logger.warning("FNHEAD4: lm_head is not a 2-D FP8 block-scaled TP1 head; keeping the FP8 path.")
        return False
    bn, bk = (int(x) for x in getattr(head, "weight_block_size", [128, 128]))
    n, k = w.shape
    with torch.no_grad():
        amax = max(float(_fp8_rows(head, r, min(r + _CHUNK, n), bn, bk).abs().amax()) for r in range(0, n, _CHUNK))
        g = amax / (448.0 * 6.0) or 1.0
        qs, ss = [], []
        for r in range(0, n, _CHUNK):
            q, sc = _quant(_fp8_rows(head, r, min(r + _CHUNK, n), bn, bk), g)
            qs.append(q); ss.append(sc)
        q = torch.cat(qs); sc = torch.cat(ss)
    mib = lambda t: t.numel() * t.element_size() / 2**20
    logger.info("FNHEAD4 target lm_head NVFP4 built: %d x %d, FP8 %.0f MiB -> NVFP4 %.0f MiB per verify", n, k,
                mib(w) + mib(s), mib(q) + mib(sc))
    return {"q": q, "s": sc, "g": g}


def target_logits(model, hidden_states: torch.Tensor):
    st = getattr(model, "_fn_head4", None)
    if st is None:
        st = build(model)
        model._fn_head4 = st
    if not st:
        return None
    from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import nvfp4_rows_gemv
    lp = model.logits_processor
    logits = nvfp4_rows_gemv(hidden_states.to(torch.bfloat16), st["q"], st["s"], st["g"]).to(hidden_states.dtype)
    logits = logits[..., : lp.org_vocab_size]
    if lp.soft_cap is not None:
        logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        logits = logits * lp.scale
    return logits
