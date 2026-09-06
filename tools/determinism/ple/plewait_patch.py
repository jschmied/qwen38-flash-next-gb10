"""PLEWAIT probe (VLLM_PLE_CHECK=1): inside vllm::ple_offload_wait's impl, log the flag value observed BEFORE the wait is
enqueued (item() syncs the current stream first, skipped while a stream is capturing) plus the current stream id. `off` removes."""
import os, sys
TARGET = os.environ.get("VLLM_POL_PY", "/opt/llm/runtime/vllm-venv-fnmain2/lib/python3.12/site-packages/vllm/model_executor/layers/ple_offload_layer.py")
ANCHOR = '''    """Wait for the CPU result without releasing its output buffer."""
    stream = torch.cuda.current_stream()
'''
NEW = '''    """Wait for the CPU result without releasing its output buffer."""
    stream = torch.cuda.current_stream()
    # ---- PLEWAIT probe (jschmied 2026-09-06), gated on VLLM_PLE_CHECK ----
    import os as _os
    if _os.environ.get("VLLM_PLE_CHECK") and not torch.cuda.is_current_stream_capturing():
        import logging as _lg
        _flag = int(sem_flag_tensor.item())
        _lg.getLogger("vllm").warning("PLEWAIT flag_before_wait=%d stream=%s tokens=%d",
                                      _flag, hex(stream.cuda_stream), int(hidden_states.shape[0]))
    # ---- end PLEWAIT probe ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  plewait probe not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  plewait probe REMOVED")
else:
    if "PLEWAIT probe" in s: print("  plewait probe already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  plewait probe INSTALLED (inert unless VLLM_PLE_CHECK=1)")
