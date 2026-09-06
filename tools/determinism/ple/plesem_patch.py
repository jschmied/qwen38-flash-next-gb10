"""PLESEM trace (VLLM_PLE_CHECK=1): log every CpuGpuSemaphore.reset/signal/wait_reset enqueue and every ple_offload_wait
with pid, stream, semaphore ptr and the two calling frames. Both processes use this class, so the worker's ops appear too."""
import os, sys
TARGET = os.environ.get("VLLM_POL_PY", "/opt/llm/runtime/vllm-venv-fnmain2/lib/python3.12/site-packages/vllm/model_executor/layers/ple_offload_layer.py")
A_RESET = '''    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
'''
A_SIGNAL = '''    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
'''
A_WR = '''    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
'''
A_WAIT = '''    """Wait for the CPU result without releasing its output buffer."""
    stream = torch.cuda.current_stream()
'''
def probe(name, ptr_expr, indent="        "):
    return f'''{indent}_plesem("{name}", stream, {ptr_expr})
'''
HELPER = '''

# ---- PLESEM trace helper (jschmied 2026-09-06), gated on VLLM_PLE_CHECK ----
def _plesem(op, stream, ptr):
    import os as _os
    if not _os.environ.get("VLLM_PLE_CHECK"):
        return
    import inspect, logging, time
    fr = inspect.stack()
    callers = "<-".join(f.function for f in fr[2:5])
    cap = torch.cuda.is_current_stream_capturing()
    logging.getLogger("vllm").warning("PLESEM %s pid=%d t=%.3f stream=%s sem=%s capturing=%s via %s",
                                      op, _os.getpid(), time.time() % 1000, hex(stream.cuda_stream), hex(ptr), cap, callers)
# ---- end PLESEM helper ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if "PLESEM trace helper" not in s: print("  plesem not installed"); raise SystemExit
    for a, nm, pe in ((A_RESET, "reset", "self._flag_tensor.data_ptr()"), (A_SIGNAL, "signal", "self._flag_tensor.data_ptr()"), (A_WR, "wait_reset", "self._flag_tensor.data_ptr()")):
        s = s.replace(a + probe(nm, pe), a)
    s = s.replace(A_WAIT + probe("gpu_wait", "sem_flag_tensor.data_ptr()", "    "), A_WAIT)
    i = s.find(HELPER); s = s[:i] + s[i+len(HELPER):]
    open(TARGET, "w").write(s); print("  plesem REMOVED")
else:
    if "PLESEM trace helper" in s: print("  plesem already installed"); raise SystemExit
    for a in (A_RESET, A_SIGNAL, A_WR, A_WAIT): assert s.count(a) == 1, ("anchor", a[:50])
    s = s.replace(A_RESET, A_RESET + probe("reset", "self._flag_tensor.data_ptr()"))
    s = s.replace(A_SIGNAL, A_SIGNAL + probe("signal", "self._flag_tensor.data_ptr()"))
    s = s.replace(A_WR, A_WR + probe("wait_reset", "self._flag_tensor.data_ptr()"))
    s = s.replace(A_WAIT, A_WAIT + probe("gpu_wait", "sem_flag_tensor.data_ptr()", "    "))
    # helper must be defined before first use at import time? it's only called at runtime -> append at end of module
    s = s + HELPER
    open(TARGET, "w").write(s); print("  plesem INSTALLED (inert unless VLLM_PLE_CHECK=1)")
