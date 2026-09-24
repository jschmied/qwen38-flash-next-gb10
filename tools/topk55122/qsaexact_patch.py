"""#55122 bench arm switch at the QSA top-k site of the main venv (ops/qsa_indexer.py, `_topk`).
  VLLM_QSA_EXACT_TOPK=1   masked torch.topk over the visible logits, -1 fill (the exact-top-k workaround)
Also logs the path taken, via the vLLM logger (worker prints get lost on main -- memory
proc-environ-invalid-for-vllm): `QSATOPK env path=...` at import, `QSATOPK call path=...` on the first call.
Inert without the env var apart from those two log lines. `off` removes it byte-exactly."""
import sys
TARGET = sys.argv[2] if len(sys.argv) > 2 else (
    "/opt/llm/runtime/vllm-venv-main1ea7/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py")
ANCHOR = """    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )
"""
NEW = """    # ---- QSAEXACT (jschmied 2026-09-24, #55122 perf bench) ----
    if not getattr(_topk, "_qsapath_logged", False):
        _topk._qsapath_logged = True
        _qsa_logger.warning("QSATOPK call path=%s", _qsa_path())
    if _os.environ.get("VLLM_QSA_EXACT_TOPK"):
        _n = logits.shape[1]
        _k = min(block_topk, _n)
        _col = torch.arange(_n, device=logits.device)
        _masked = logits.masked_fill(
            _col[None, :] >= visible_blocks.to(torch.int64)[:, None], float("-inf"))
        _vals, _idx = torch.topk(_masked, _k, dim=1)
        _idx = _idx.masked_fill(torch.isinf(_vals), -1).to(block_indices.dtype)
        if _k < block_topk:
            block_indices.fill_(-1)
        block_indices[:, :_k].copy_(_idx)
    else:
        topk_op(
            logits,
            visible_blocks,
            block_indices,
            topk_workspace,
            block_topk,
            logits.shape[1],
        )
    # ---- end QSAEXACT ----
"""
HEAD_ANCHOR = "_TOPK_WORKSPACE_BYTES = 1024 * 1024\n"
HEAD_NEW = HEAD_ANCHOR + """
# ---- QSAEXACT head (jschmied 2026-09-24, #55122 perf bench) ----
import os as _qsa_os
from vllm.logger import init_logger as _qsa_init_logger
_qsa_logger = _qsa_init_logger(__name__)


def _qsa_path():
    if _qsa_os.environ.get("VLLM_QSA_EXACT_TOPK"):
        return "exact"
    if _qsa_os.environ.get("VLLM_QSA_DET_TOPK"):
        return "det:" + _qsa_os.environ.get("VLLM_QSA_DET_LIB", "/opt/llm/kernel-det/_C_det.so")
    return "stock"


_qsa_logger.warning("QSATOPK env path=%s", _qsa_path())
# ---- end QSAEXACT head ----
"""
s = open(TARGET).read()
if sys.argv[1:2] == ["off"]:
    if "QSAEXACT" not in s:
        print("  qsaexact not installed"); raise SystemExit
    s = s.replace(NEW, ANCHOR).replace(HEAD_NEW, HEAD_ANCHOR)
    assert "QSAEXACT" not in s, "partial removal"
    open(TARGET, "w").write(s); print("  qsaexact REMOVED")
else:
    if "QSAEXACT" in s:
        print("  qsaexact already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1 and s.count(HEAD_ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW).replace(HEAD_ANCHOR, HEAD_NEW))
    print("  qsaexact INSTALLED (inert unless VLLM_QSA_EXACT_TOPK=1; logs QSATOPK path lines)")
