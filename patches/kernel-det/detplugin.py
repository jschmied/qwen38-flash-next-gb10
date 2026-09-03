# pytest plugin: run the upstream test file against the standalone _C_det build
# (the venv's _C.persistent_topk is the stock kernel). Usage:
#   PYTHONSAFEPATH=1 python -m pytest -p detplugin test_upstream_top_k_per_row.py -k persistent_topk_
import os, torch
torch.ops.load_library(os.environ["DET_SO"])
import vllm  # noqa: F401
def pytest_collection_modifyitems(session, config, items):
    for it in items:
        mod = it.module
        if hasattr(mod, "_run_persistent_topk") and not getattr(mod, "_det_swapped", False):
            def _run_det(logits, lengths, top_k, _ws=mod.RADIX_TOPK_WORKSPACE_SIZE):
                indices = torch.empty((logits.shape[0], top_k), dtype=torch.int32, device="cuda")
                workspace = torch.empty(_ws, dtype=torch.uint8, device="cuda")
                torch.ops._C_det.persistent_topk(logits, lengths, indices, workspace, top_k, logits.shape[1])
                torch.cuda.synchronize(); return indices
            mod._run_persistent_topk = _run_det; mod._det_swapped = True
            print("DETPLUGIN: _run_persistent_topk -> _C_det", flush=True)
