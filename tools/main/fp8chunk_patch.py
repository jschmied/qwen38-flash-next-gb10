"""M-chunking for the CUTLASS blockwise-FP8 GEMM on sm_12x (finding 71: on GB10's 24 MiB L2 the
128x128 config drops 163 -> 95 -> 51 TFLOPS from 4k to 16k rows; 4,096-row chunks recover 3x).
Env: VLLM_FP8_BLOCK_M_CHUNK (rows; default 4096 on capability family 120, 0 = off elsewhere).
Target: vllm/model_executor/kernels/linear/scaled_mm/cutlass.py of the running interpreter. `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_CUTLASS_PY"): return os.environ["VLLM_CUTLASS_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "model_executor/kernels/linear/scaled_mm/cutlass.py")
TARGET = _target()
ANCHOR = '''    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )
'''
NEW = '''    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        # ---- FP8CHUNK (jschmied 2026-09-03): on sm_12x (24 MiB L2) the blockwise kernel
        # re-streams the weight per tile row once M spans many tile rows; chunking M at
        # 4,096 rows keeps it at ~160 TFLOPS instead of 95 (8k) / 51 (16k). ----
        chunk = _fp8_block_m_chunk()
        M = A.shape[0]
        if chunk and M > chunk:
            out = torch.empty((M, B.shape[0]), dtype=out_dtype, device=A.device)
            for i in range(0, M, chunk):
                out[i : i + chunk] = ops.cutlass_scaled_mm(
                    A[i : i + chunk],
                    B.T,
                    out_dtype=out_dtype,
                    scale_a=As[i : i + chunk],
                    scale_b=Bs.T,
                )
            return out
        # ---- end FP8CHUNK ----
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


_FP8_BLOCK_M_CHUNK: int | None = None


def _fp8_block_m_chunk() -> int:
    global _FP8_BLOCK_M_CHUNK
    if _FP8_BLOCK_M_CHUNK is None:
        import os as _os
        v = _os.environ.get("VLLM_FP8_BLOCK_M_CHUNK")
        if v is not None:
            _FP8_BLOCK_M_CHUNK = int(v)
        else:
            from vllm.platforms import current_platform
            _FP8_BLOCK_M_CHUNK = (
                4096
                if current_platform.is_cuda()
                and current_platform.is_device_capability_family(120)
                else 0
            )
        print(f"FP8CHUNK active: M chunk = {_FP8_BLOCK_M_CHUNK}", flush=True)
    return _FP8_BLOCK_M_CHUNK
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  fp8chunk not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  fp8chunk REMOVED")
else:
    if "FP8CHUNK" in s: print("  fp8chunk already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  fp8chunk INSTALLED in", TARGET)
