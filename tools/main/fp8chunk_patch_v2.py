"""M-chunking for the CUTLASS blockwise-FP8 GEMM on sm_12x (finding 71: on GB10's 24 MiB L2 the
128x128 config drops 163 -> 95 -> 51 TFLOPS from 4k to 16k rows; 4,096-row chunks recover 3x).
Env: VLLM_FP8_BLOCK_M_CHUNK (rows; default 4096 on capability family 120, 0 = off elsewhere).
Target: vllm/model_executor/kernels/linear/scaled_mm/cutlass.py of the running interpreter. `off` removes.
v2 (2026-09-04): loop inside an opaque torch.library custom op; env resolved + activation line at import."""
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
        # ---- FP8CHUNK v2 (jschmied 2026-09-04): on sm_12x (24 MiB L2) the blockwise kernel
        # re-streams the weight per tile row once M spans many tile rows; chunking M at
        # 4,096 rows keeps it at ~160 TFLOPS instead of 95 (8k) / 51 (16k). The loop is an
        # opaque custom op so torch.compile neither traces nor specialises it on M. ----
        if _FP8_BLOCK_M_CHUNK and A.shape[0] > _FP8_BLOCK_M_CHUNK:
            return torch.ops.fp8chunk.scaled_mm_chunked(
                A, B, As, Bs, _FP8_BLOCK_M_CHUNK, out_dtype == torch.bfloat16
            )
        # ---- end FP8CHUNK ----
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


def _fp8chunk_resolve() -> int:
    import os as _os
    v = _os.environ.get("VLLM_FP8_BLOCK_M_CHUNK")
    if v is not None:
        n = int(v)
    else:
        from vllm.platforms import current_platform
        n = (
            4096
            if current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            else 0
        )
    print(f"FP8CHUNK active: M chunk = {n}", flush=True)
    return n


_FP8_BLOCK_M_CHUNK: int = _fp8chunk_resolve()


@torch.library.custom_op("fp8chunk::scaled_mm_chunked", mutates_args=())
def _fp8chunk_scaled_mm_chunked(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, chunk: int, bf16: bool
) -> torch.Tensor:
    out_dtype = torch.bfloat16 if bf16 else torch.float16
    M = A.shape[0]
    out = torch.empty((M, B.shape[0]), dtype=out_dtype, device=A.device)
    for i in range(0, M, chunk):
        # The kernel deduces the activation-scale layout from its own M (column-major,
        # M fastest): a row slice of the full tensor is silently WRONG; re-materialise.
        As_c = As[i : i + chunk].t().contiguous().t()
        out[i : i + chunk] = ops.cutlass_scaled_mm(
            A[i : i + chunk].contiguous(), B.T, out_dtype=out_dtype, scale_a=As_c, scale_b=Bs.T
        )
    return out


@_fp8chunk_scaled_mm_chunked.register_fake
def _(A, B, As, Bs, chunk, bf16):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16 if bf16 else torch.float16)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  fp8chunk not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  fp8chunk REMOVED")
else:
    if "FP8CHUNK" in s: print("  fp8chunk already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  fp8chunk INSTALLED in", TARGET)
