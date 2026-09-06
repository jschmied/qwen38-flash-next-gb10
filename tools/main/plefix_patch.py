"""PLEFIX (candidate fix for vllm#53899): reset every PLE offload semaphore on the model stream BEFORE a real request is
submitted, so the GPU-side wait can only be satisfied by the copy for this step's inputs. `off` removes."""
import os, sys
TARGET = os.environ.get("VLLM_CONN_PY", "/opt/llm/runtime/vllm-venv-fnmain2/lib/python3.12/site-packages/vllm/v1/ple_offload/connector.py")
ANCHOR = '''        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        self._launch(num_reqs, num_tokens)
'''
NEW = '''        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        # Clear any semaphore still raised by a dummy or capture forward before
        # the CPU worker can raise this step's. capture_model() signals dummy
        # outputs and then runs real steps through execute_model: the first
        # real wait would otherwise pass on the dummy signal, its release would
        # reset, and the worker's copy for that step would raise the flag for
        # the *next* step -- every step then consumes the previous step's rows.
        # The reset is per rank (each TP rank owns its buffer and semaphore),
        # so it runs before the tp_rank check in _launch.
        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)
        self._launch(num_reqs, num_tokens)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  plefix not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  plefix REMOVED")
else:
    if NEW in s: print("  plefix already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  plefix INSTALLED (reset-before-launch)")
