"""Dense(r) DRAFTER: give the MTP layer's QSA indexer its own budget (env VLLM_DRAFT_QSA_BUDGET),
leaving the target's 2048 untouched. With a budget >= context the drafter attends to every block,
which is what llama.cpp's MTP head does by design. Needs the exact top-k patch (persistent_topk
only accepts k=512/2048). Inert without the env var; `off` restores byte-exactly."""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
ANCHOR = '''        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
'''
NEW = ANCHOR + '''        _b = __import__("os").environ.get("VLLM_DRAFT_QSA_BUDGET")  # DENSEDRAFT (jschmied 2026-09-03)
        if _b:
            import copy as _copy
            _mc = _copy.copy(draft_vllm_config.model_config)
            _hf = _copy.deepcopy(_mc.hf_config)
            _mc.hf_config = _hf
            _tc = _hf.get_text_config() if hasattr(_hf, "get_text_config") else _hf
            _tc.indexer_budget = int(_b)
            try:
                _mc.hf_text_config = _tc
            except AttributeError:
                pass
            draft_vllm_config.model_config = _mc
            print(f"DENSEDRAFT: drafter indexer_budget={_tc.indexer_budget} (target {vllm_config.model_config.hf_text_config.indexer_budget})", flush=True)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if "DENSEDRAFT" not in s: print("  densedraft not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  densedraft REMOVED")
else:
    if "DENSEDRAFT" in s: print("  densedraft already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  densedraft INSTALLED (inert unless VLLM_DRAFT_QSA_BUDGET is set)")
