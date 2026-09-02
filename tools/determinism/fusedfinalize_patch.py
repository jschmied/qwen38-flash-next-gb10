#!/usr/bin/env python3
"""THE FIX CANDIDATE: pass use_fused_finalize=False to FlashInfer's cutlass MoE.

FlashInfer documents it (fused_moe/core.py): "The fused epilogue reduces expert outputs via
non-associative atomics, so results are not deterministic run-to-run. Set to False to use the
non-fused, deterministic finalize path." vLLM never passes the argument, so the atomic path is what
every arm ran. Findings 26-30 located the divergence to exactly this kernel.
Env-gated on VLLM_MOE_DET_FINALIZE=1 so one venv serves both arms. Usage: [off]
"""
import sys, re
T = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/"
     "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py")
MARK = "# DET FINALIZE (jschmied 2026-09-02)"
s = open(T).read()
anchor = "            use_w4_group_scaling=use_w4_group_scaling,\n        )"
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if MARK not in s: print("  det-finalize not installed"); sys.exit(0)
    s2 = re.sub(r"\n[ \t]*use_fused_finalize=[^\n]*\n[ \t]*profile_ids=[^\n]*" + re.escape(MARK) + r"\n", "\n", s, count=1)
    assert MARK not in s2, "removal failed"; open(T, "w").write(s2); print("  det-finalize REMOVED"); sys.exit(0)
if MARK in s: print("  det-finalize already installed"); sys.exit(0)
assert s.count(anchor) == 1, f"call-site anchor count {s.count(anchor)}"
ins = ('            use_w4_group_scaling=use_w4_group_scaling,\n'
       '            use_fused_finalize=not bool(__import__("os").environ.get("VLLM_MOE_DET_FINALIZE")),\n            profile_ids=([-1, -1] if __import__("os").environ.get("VLLM_MOE_DET_FINALIZE") else None),  ' + MARK + '\n'
       '        )')
open(T, "w").write(s.replace(anchor, ins, 1)); print("  det-finalize INSTALLED (inert unless VLLM_MOE_DET_FINALIZE=1)")
