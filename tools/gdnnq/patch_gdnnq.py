"""FNGDNNQ overlay: copy gdn_norm_quant.py next to the GDN layer (model_executor/layers/mamba/gdn/fn_gdn_norm_quant.py)
and hook QwenGatedDeltaNetAttention._output_projection (env FN_GDNNQ=1, read at import like FNHCFUSE; no caching in the
traced forward). Usage: python patch_gdnnq.py <site-packages>/vllm [off]. Backup *.orig-gdnnq."""
import os, shutil, sys
pkg = sys.argv[1]; off = len(sys.argv) > 2 and sys.argv[2] == "off"
F = os.path.join(pkg, "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
OPS = os.path.join(pkg, "model_executor/layers/mamba/gdn/fn_gdn_norm_quant.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_norm_quant.py")
M = "# FNGDNNQ"
if off:
    shutil.copy2(F + ".orig-gdnnq", F)
    if os.path.exists(OPS): os.remove(OPS)
    print("gdnnq off", M in open(F).read()); sys.exit(0)
s = open(F).read()
if M in s: print("already on"); sys.exit(0)
if not os.path.exists(F + ".orig-gdnnq"): shutil.copy2(F, F + ".orig-gdnnq")
a0 = "from vllm.model_executor.layers.layernorm import RMSNormGated\n"
b0 = (a0 + "import os as _fn_os  " + M + "\n"
      "_FN_GDNNQ = _fn_os.environ.get(\"FN_GDNNQ\") == \"1\"  " + M + "\n"
      "if _FN_GDNNQ:  " + M + "\n"
      "    from .fn_gdn_norm_quant import gdn_output_fusable, gdn_output_fused  " + M + "\n")
a1 = ('        core_attn_out = self.norm(core_attn_out, z)\n'
      '        output, _ = self.out_proj(core_attn_out.flatten(-2))\n'
      '        return output\n')
b1 = ('        if _FN_GDNNQ and gdn_output_fusable(self):  ' + M + '\n'
      '            return gdn_output_fused(self, core_attn_out, z)  ' + M + '\n' + a1)
for a, b in ((a0, b0), (a1, b1)):
    assert s.count(a) == 1, (a[:50], s.count(a))
    s = s.replace(a, b)
compile(s, F, "exec")
shutil.copy2(SRC, OPS); open(F, "w").write(s); print("gdnnq on", s.count(M))
