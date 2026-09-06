"""PLEFLAG probe (VLLM_PLE_CHECK=1) in the model runner: flag value (a) at execute_model entry before prepare_forward,
(b) right after release_outputs (after a stream sync), (c) at the end of capture_model; plus stream identities. `off` removes."""
import os, sys
TARGET = os.environ.get("VLLM_MR_PY", "/opt/llm/runtime/vllm-venv-fnmain2/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")
A1 = '''        if self._ple_offload_connector is not None:
            self._ple_offload_connector.prepare_forward(
'''
N1 = '''        # ---- PLEFLAG probe a (jschmied 2026-09-06), gated on VLLM_PLE_CHECK ----
        import os as _os
        if (self._ple_offload_connector is not None and not dummy_run and _os.environ.get("VLLM_PLE_CHECK")
                and not torch.cuda.is_current_stream_capturing()):
            torch.cuda.synchronize()
            _fl = {n: int(l._sem.flag_tensor.item()) for n, l in self._ple_offload_connector._layers.items()}
            logger.warning("PLEFLAG entry tokens=%d flags=%s cur=%s default=%s copy=%s", input_batch.num_tokens,
                           _fl, hex(torch.cuda.current_stream(self.device).cuda_stream),
                           hex(torch.cuda.default_stream(self.device).cuda_stream), hex(self.output_copy_stream.cuda_stream))
        # ---- end PLEFLAG a ----
''' + A1
A2 = '''        if self._ple_offload_connector is not None:
            self._ple_offload_connector.release_outputs()

        if self.is_last_pp_rank:
'''
N2 = '''        if self._ple_offload_connector is not None:
            self._ple_offload_connector.release_outputs()
        # ---- PLEFLAG probe b ----
        if (self._ple_offload_connector is not None and not dummy_run and _os.environ.get("VLLM_PLE_CHECK")
                and not torch.cuda.is_current_stream_capturing()):
            torch.cuda.synchronize()
            _fl = {n: int(l._sem.flag_tensor.item()) for n, l in self._ple_offload_connector._layers.items()}
            logger.warning("PLEFLAG after_release tokens=%d flags=%s", input_batch.num_tokens, _fl)
        # ---- end PLEFLAG b ----

        if self.is_last_pp_rank:
'''
A3 = '''            if self._ple_offload_connector is not None and capture_decoder:
                self._ple_offload_connector.release_outputs()
'''
N3 = '''            if self._ple_offload_connector is not None and capture_decoder:
                self._ple_offload_connector.release_outputs()
        # ---- PLEFLAG probe c ----
        import os as _os2
        if self._ple_offload_connector is not None and _os2.environ.get("VLLM_PLE_CHECK"):
            torch.cuda.synchronize()
            _fl = {n: int(l._sem.flag_tensor.item()) for n, l in self._ple_offload_connector._layers.items()}
            logger.warning("PLEFLAG end_of_capture_model flags=%s cur=%s", _fl, hex(torch.cuda.current_stream(self.device).cuda_stream))
        # ---- end PLEFLAG c ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if "PLEFLAG probe" not in s: print("  pleflag not installed"); raise SystemExit
    for n, a in ((N1, A1), (N2, A2), (N3, A3)):
        assert s.count(n) == 1, "revert anchor"; s = s.replace(n, a)
    open(TARGET, "w").write(s); print("  pleflag REMOVED")
else:
    if "PLEFLAG probe" in s: print("  pleflag already installed"); raise SystemExit
    for a in (A1, A2, A3): assert s.count(a) == 1, ("anchor", a[:60])
    s = s.replace(A1, N1).replace(A2, N2).replace(A3, N3)
    open(TARGET, "w").write(s); print("  pleflag INSTALLED (inert unless VLLM_PLE_CHECK=1)")
