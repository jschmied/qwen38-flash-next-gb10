# Run moe_ncu.py against a JIT-built fused_moe_120 from a COPIED csrc tree (FN_FI_CSRC), bypassing the AOT jit-cache package; prints a checksum.
import os, sys, pathlib, importlib
import flashinfer.jit.env as E
c=os.environ.get("FN_FI_CSRC")
if c:
    for modname in ("flashinfer.jit.env","flashinfer.jit.core","flashinfer.jit.fused_moe"):
        m=importlib.import_module(modname)
        if hasattr(m,"FLASHINFER_CSRC_DIR"): m.FLASHINFER_CSRC_DIR=pathlib.Path(c)
        if hasattr(m,"FLASHINFER_AOT_DIR"): m.FLASHINFER_AOT_DIR=pathlib.Path("/nonexistent-aot")
    print("JIT from", c, "swz", os.environ.get("FN_MOE_SWZ","1"), "raster", os.environ.get("FN_MOE_RASTER","N"), flush=True)
else:
    print("AOT module (baseline)", flush=True)
ns={"__name__":"__main__"}; exec(compile(open("/opt/llm/runners/moe_ncu.py").read(),"moe_ncu.py","exec"), ns)
o=ns["out"]; print(f"checksum {o.float().abs().sum().item():.7e} max={o.float().abs().max().item():.5e} nan={bool(o.isnan().any())}", flush=True)
