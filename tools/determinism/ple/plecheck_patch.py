"""PLECHECK: env-gated (VLLM_PLE_CHECK=1) diagnostic in execute_model — after the forward and before the PLE outputs are
released, hash every PLE layer's GPU output buffer and count its non-zero rows. A zero buffer at a real step means the model
consumed the dummy (zeroed) content, i.e. the wait did not hold. `off` removes it."""
import os, sys
TARGET = os.environ.get("VLLM_MR_PY", "/opt/llm/runtime/vllm-venv-fnmain2/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")
ANCHOR = '''        # The model has consumed every PLE output, so the CPU worker may reuse
        # the registered buffers for the next request.
        if self._ple_offload_connector is not None:
            self._ple_offload_connector.release_outputs()
'''
NEW = '''        # ---- PLECHECK (jschmied 2026-09-06), gated on VLLM_PLE_CHECK ----
        import os as _os
        if (self._ple_offload_connector is not None and not dummy_run
                and _os.environ.get("VLLM_PLE_CHECK")):
            import hashlib as _hl
            torch.cuda.synchronize()
            _n = input_batch.num_tokens
            for _name, _layer in self._ple_offload_connector._layers.items():
                _b = _layer._gpu_output_buffer[:_n].float()
                _nz = int((_b.abs().sum(dim=-1) > 0).sum().item())
                _h = _hl.sha256(_b.cpu().numpy().tobytes()).hexdigest()[:10]
                logger.warning("PLECHECK tokens=%d layer=%s nonzero_rows=%d absmean=%.4e hash=%s",
                               _n, _name, _nz, float(_b.abs().mean().item()), _h)
        # ---- end PLECHECK ----
''' + ANCHOR
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  plecheck not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  plecheck REMOVED")
else:
    if "PLECHECK" in s: print("  plecheck already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  plecheck INSTALLED (inert unless VLLM_PLE_CHECK=1)")
