import torch, flashinfer.fused_moe.core as core
# the JIT module (cached build; no rebuild expected) -- the one MoERunner closes over for sm_121
mod = core.gen_cutlass_fused_moe_sm120_module(False).build_and_load()
print("  module attrs:", [a for a in dir(mod) if not a.startswith("_")][:12])
x, w, o = torch.bfloat16, torch.int64, torch.bfloat16  # isNvfp4Quant(): weight dtype int64
for fused in (True, False):
    r = mod.init(x, w, o, False, False, False, False, fused)
    g1, g2, tot = r.get_gemm1_tactic_count(), r.get_gemm2_tactic_count(), r.get_tactic_num()
    print(f"  use_fused_finalize={fused!s:<5}  gemm1_tactics={g1:<4} gemm2_tactics={g2:<4} total={tot}")
