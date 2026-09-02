"""Two env-gated candidate fixes at the QSA block selection site (ops/qsa.py):
  VLLM_QSA_EXACT_TOPK=1  exact torch.topk over the visible logits instead of persistent_topk,
                         canonical ascending order, -1 padding trailing
  VLLM_QSA_SORT=1        keep persistent_topk, canonicalise its order (padding stays trailing)
Inert without the env vars. `off` removes byte-exactly."""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
ANCHOR = "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
NEW = '''        # ---- QSAFIX2 (jschmied 2026-09-02) ----
        import os as _os
        _qsafix = _os.environ.get("VLLM_QSA_EXACT_TOPK") and "exact" or (_os.environ.get("VLLM_QSA_SORT") and "sort") or ""
        if _qsafix == "exact":
            _vis = visible_blocks.to(torch.int64)
            _col = torch.arange(logits.shape[1], device=logits.device)
            _masked = logits.float().masked_fill(_col[None, :] >= _vis[:, None], float("-inf"))
            if _os.environ.get("VLLM_QSA_EXACT_TOPK") == "stable":
                # ties broken by index: stable descending sort, take the first k
                _sv, _si = torch.sort(_masked, dim=1, descending=True, stable=True)
                _vals, _idx = _sv[:, :block_topk], _si[:, :block_topk]
            else:
                _vals, _idx = torch.topk(_masked, block_topk, dim=1)
            _idx = _idx.masked_fill(torch.isinf(_vals), -1)
            _key = _idx.masked_fill(_idx < 0, 2**31 - 1).sort(dim=1).values
            blocks.copy_(_key.masked_fill(_key == 2**31 - 1, -1).to(torch.int32))
        else:
            topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)
            if _qsafix == "sort":
                _key = blocks.masked_fill(blocks < 0, 2**31 - 1).sort(dim=1).values
                blocks.copy_(_key.masked_fill(_key == 2**31 - 1, -1))
        if _qsafix and not getattr(qsa_select_paged_tokens, "_qsafix_logged", False):
            qsa_select_paged_tokens._qsafix_logged = True
            print(f"QSAFIX2 active: {_qsafix} mode={_os.environ.get('VLLM_QSA_EXACT_TOPK','')}", flush=True)
        # ---- end QSAFIX2 ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  qsafix2 not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  qsafix2 REMOVED")
else:
    if "QSAFIX2" in s: print("  qsafix already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  qsafix2 INSTALLED (inert unless VLLM_QSA_EXACT_TOPK=1 or VLLM_QSA_SORT=1)")
