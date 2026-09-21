#!/usr/bin/env python3
"""Fix for vllm#50189: mask the -1 "not routed" sentinel before the b12x MoE launch.

vLLM uses topk_ids == -1 to mean "this (token, slot) is not routed" -- see
fused_moe.py:167 / :424 and deep_gemm_utils.py:109, where the Triton backends branch on it.
FlashInferB12xExperts.apply() forwards topk_ids unmasked, and the b12x kernel indexes its
per-expert state with them, so a negative id is an out-of-bounds access -> Xid 31 MMU fault
(ACCESS_TYPE_VIRT_WRITE). The profile run in determine_available_memory() emits an all -1
routing table, which is why the engine dies at startup with no traffic.

Route invalid slots to expert 0 with weight 0: numerically identical to skipping them, and
branch-free (two elementwise kernels over topk_ids, no host sync)."""
import ast, pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
s = src.read_text()
if "FIX-50189" in s:
    print("  already patched"); sys.exit(0)

old = """        wrapper_output = wrapper.run(
            x=hidden_states,"""
new = """        # FIX-50189: -1 is vLLM's "not routed" sentinel (fused_moe.py:167). The b12x kernel
        # indexes per-expert state with topk_ids, so forwarding a negative id writes out of
        # bounds. Send those slots to expert 0 with zero weight -- same result as skipping
        # them, and no host sync (an .any() check here would stall the stream).
        _ids = topk_ids.to(torch.int32)
        _valid = _ids >= 0
        _safe_ids = torch.where(_valid, _ids, torch.zeros_like(_ids))
        _safe_weights = topk_weights * _valid.to(topk_weights.dtype)

        wrapper_output = wrapper.run(
            x=hidden_states,"""
assert s.count(old) == 1, f"anchor count {s.count(old)}"
s = s.replace(old, new, 1)

# point the call at the masked tensors
o2 = """            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,"""
n2 = """            token_selected_experts=_safe_ids,
            token_final_scales=_safe_weights,"""
assert s.count(o2) == 1, f"call anchor count {s.count(o2)}"
s = s.replace(o2, n2, 1)
ast.parse(s)
dst.write_text(s)
print("  patched copy written, parses OK")
