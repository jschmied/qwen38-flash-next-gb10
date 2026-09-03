"""Env-gated dump of the QSA block selection (preview tree, ops/qsa.py): with VLLM_QSA_DUMP=<dir> the first
16 selection calls write `blocks` [rows, block_topk] (int32, -1 padded) and `visible_blocks` [rows] to
<dir>/sel_<n>.pt. For the prefill index-sharing / tile-union question (prefill-plan §5). `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_QSA_PY"): return os.environ["VLLM_QSA_PY"]
    import vllm; root=os.path.dirname(vllm.__file__)
    for rel in ("models/qwen3_8_flash_next/nvidia/ops/qsa.py","models/qwen4_exp/nvidia/ops/qsa.py"):
        if os.path.exists(os.path.join(root,rel)): return os.path.join(root,rel)
    raise SystemExit("qsa.py not found")
TARGET=_target()
ANCHOR="        expand_qsa_block_indices_cuda(\n"
NEW='''        # ---- QSADUMP (jschmied 2026-09-03) ----
        import os as _os
        if _os.environ.get("VLLM_QSA_DUMP"):
            _n = getattr(qsa_select_paged_tokens, "_qsadump_n", 0)
            if _n < 16:
                qsa_select_paged_tokens._qsadump_n = _n + 1
                torch.save({"blocks": blocks.cpu(), "visible_blocks": visible_blocks.cpu(), "columns": int(columns)},
                           _os.path.join(_os.environ["VLLM_QSA_DUMP"], f"sel_{_n}.pt"))
        # ---- end QSADUMP ----
        expand_qsa_block_indices_cuda(
'''
s=open(TARGET).read()
if sys.argv[1:] and sys.argv[1]=="off":
    if NEW not in s: print("  qsadump not installed"); raise SystemExit
    open(TARGET,"w").write(s.replace(NEW,ANCHOR)); print("  qsadump REMOVED")
else:
    if "QSADUMP" in s: print("  qsadump already installed"); raise SystemExit
    assert s.count(ANCHOR)==1, "anchor"
    open(TARGET,"w").write(s.replace(ANCHOR,NEW)); print("  qsadump INSTALLED in", TARGET)
