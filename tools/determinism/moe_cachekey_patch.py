#!/usr/bin/env python3
"""FlashInfer fix: make the cutlass MoERunner's autotune file-cache key include the parameters
that change its tactic table.

flashinfer#3367 (0.6.13) made the persistent autotune cache key (custom_op, runner_class, profile,
extras) and added get_cache_key_extras() to TrtllmGemmRunner only. The cutlass fused-MoE
MoERunner never got one, so a runner built with use_fused_finalize=False shares cache entries with
the fused runner and is handed fused-range GEMM2 tactic ids (40 fused vs 20 non-fused tactics on
sm_121) -> "Invalid gemm2 profile id". Pure correctness: widening the key can only prevent a wrong
hit. Not env-gated. Usage: moe_cachekey_patch.py [off]
"""
import sys, re
T = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/flashinfer/fused_moe/core.py")
MARK = "# CACHE-KEY EXTRAS (jschmied 2026-09-02, cf. flashinfer#3367)"
s = open(T).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if MARK not in s: print("  cache-key extras not installed"); sys.exit(0)
    s2 = re.sub(r"\n[ \t]*" + re.escape(MARK) + r".*?\n(?=[ \t]*def get_valid_tactics\()", "\n", s, count=1, flags=re.S)
    assert MARK not in s2, "removal failed"; open(T, "w").write(s2); print("  cache-key extras REMOVED"); sys.exit(0)
if MARK in s: print("  cache-key extras already installed"); sys.exit(0)
anchor = re.search(r"^([ \t]*)def get_valid_tactics\(\s*\n", s, re.M)
assert anchor, "get_valid_tactics not found in MoERunner"
ind = anchor.group(1)
ins = (f"{ind}{MARK}\n"
       f"{ind}def get_cache_key_extras(self, inputs):\n"
       f"{ind}    # every field that changes the runner's tactic table must be in the file-cache key\n"
       f"{ind}    return (\n"
       f"{ind}        self.use_fused_finalize, self.use_deepseek_fp8_block_scale,\n"
       f"{ind}        self.use_w4_group_scaling, self.use_mxfp8_act_scaling, self.use_packed_weights,\n"
       f"{ind}        self.min_latency_mode, str(self.activation_type),\n"
       f"{ind}    )\n\n")
s = s[:anchor.start()] + ins + s[anchor.start():]
open(T, "w").write(s); print("  cache-key extras INSTALLED on MoERunner")
