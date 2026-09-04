"""Drop-in for the shipped preview image (and main): pad M to a multiple of 4 inside the sm_12x blockwise-FP8
GEMM apply so the kernel never takes the `swap_ab = (M<=64)||(M%4!=0)` slow path (upstream fix #52775 is C++,
merged 2026-08-19, absent from the preview). Measured on the preview: TTFT 8k 4.52 -> 2.86 s at batch 8192,
3.05 -> 2.81 s at batch 4096 (finding 69/70), for a one-row pad. Zero-row pad of A + unit scale rows, output
sliced; the whole call incl. the M test is an opaque custom op (v2): vLLM's compile wrapper freezes Python branches at trace time. Env: VLLM_FP8_PAD_M4
(1 = on [default on capability family 120], 0 = off). Target: vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
of the running interpreter (VLLM_CUTLASS_PY overrides). `off` removes."""
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
        # ---- FP8 M4PAD v2 (jschmied 2026-09-04): the sm_12x blockwise kernel routes M%4!=0 (and M<=64)
        # to a swap_ab path ~1.6x slower at prefill M (fixed upstream in #52775, C++). The M test MUST
        # live inside the opaque op: vLLM's compile wrapper traces once (at the profiling shape, M%4==0)
        # and then bypasses Dynamo's guards, so a Python-level branch here is frozen at trace time. ----
        if _FP8_PAD_M4:
            return torch.ops.fp8m4pad.scaled_mm_padded(A, B, As, Bs, out_dtype == torch.bfloat16)
        # ---- end FP8 M4PAD ----
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


def _fp8_m4pad_resolve() -> bool:
    import os as _os
    v = _os.environ.get("VLLM_FP8_PAD_M4")
    if v is not None:
        on = v not in ("0", "", "false", "False")
    else:
        from vllm.platforms import current_platform
        on = current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    print(f"FP8M4PAD active: {int(on)}", flush=True)
    return on


_FP8_PAD_M4: bool = _fp8_m4pad_resolve()


@torch.library.custom_op("fp8m4pad::scaled_mm_padded", mutates_args=())
def _fp8_m4pad_scaled_mm(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, bf16: bool
) -> torch.Tensor:
    out_dtype = torch.bfloat16 if bf16 else torch.float16
    M = A.shape[0]
    if M <= 64 or M % 4 == 0:  # decode shapes and aligned chunks: the kernel's own fast path
        return ops.cutlass_scaled_mm(A, B.T, out_dtype=out_dtype, scale_a=As, scale_b=Bs.T)
    pad = (-M) % 4
    A_p = torch.cat([A, A.new_zeros((pad, A.shape[1]))], dim=0)
    # The kernel derives the activation-scale layout (column-major, M fastest) from its own M:
    # re-materialise the padded scales in that layout instead of slicing/concatenating views.
    As_p = torch.cat([As, As.new_ones((pad, As.shape[1]))], dim=0).t().contiguous().t()
    out = ops.cutlass_scaled_mm(A_p, B.T, out_dtype=out_dtype, scale_a=As_p, scale_b=Bs.T)
    return out[:M]


@_fp8_m4pad_scaled_mm.register_fake
def _(A, B, As, Bs, bf16):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16 if bf16 else torch.float16)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  fp8m4pad not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  fp8m4pad REMOVED")
else:
    if "FP8M4PAD" in s: print("  fp8m4pad already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor (is fp8chunk installed? remove it first)"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  fp8m4pad INSTALLED in", TARGET)
