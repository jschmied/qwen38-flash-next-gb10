"""v2: ARM-file gate. Env-gated dump of the QSA block selection (preview tree, ops/qsa.py): with VLLM_QSA_DUMP=<dir> the first
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
NEW='''        # ---- QSADUMP v2 (jschmied 2026-09-04): dumps only once <dir>/ARM exists (the runner creates it
        # after the server is up), so warmup/profiling passes cannot use up the budget. ----
        import os as _os
        _d = _os.environ.get("VLLM_QSA_DUMP")
        if _d and _os.path.exists(_os.path.join(_d, "ARM")):
            _n = getattr(qsa_select_paged_tokens, "_qsadump_n", 0)
            if _n < 96:
                qsa_select_paged_tokens._qsadump_n = _n + 1
                torch.save({"blocks": blocks.cpu(), "visible_blocks": visible_blocks.cpu(), "columns": int(columns)},
                           _os.path.join(_d, f"sel_{_n:03d}_{int(blocks.shape[0])}.pt"))
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
