"""FNHCFUSE overlay: copy ops/hc_fused.py and hook GatedResidual.combine_and_mix (env FN_HCFUSE=1).
Usage: python patch_hcfuse.py <site-packages>/vllm [off]. On: backs up hyperconnection.py once (*.orig-hcfuse).
Off: restores the backup and removes hc_fused.py."""
import os, shutil, sys
pkg = sys.argv[1]; off = len(sys.argv) > 2 and sys.argv[2] == "off"
D = os.path.join(pkg, "models/qwen4_exp/nvidia")
HCF = os.path.join(D, "hyperconnection.py"); OPS = os.path.join(D, "ops/hc_fused.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hc_fused.py")
M = "# FNHCFUSE"
if off:
    shutil.copy2(HCF + ".orig-hcfuse", HCF)
    if os.path.exists(OPS): os.remove(OPS)
    print("hcfuse off", M in open(HCF).read()); sys.exit(0)
s = open(HCF).read()
if M in s:
    print("already on"); sys.exit(0)
if not os.path.exists(HCF + ".orig-hcfuse"):
    shutil.copy2(HCF, HCF + ".orig-hcfuse")
a1 = "import torch\nfrom torch import nn\n"
b1 = ("import os  " + M + "\n\n" + a1)
a2 = ("from .ops.hc import (\n")
b2 = ("_FN_HCFUSE = os.environ.get(\"FN_HCFUSE\") == \"1\"  " + M + "\n"
      "if _FN_HCFUSE:  " + M + "\n"
      "    from .ops.hc_fused import hc_combine_mix_fused  " + M + "\n" + a2)
a3 = ("        hidden_states, xn = hc_combine_norm(\n"
      "            hidden_states,\n"
      "            prev_block_output,\n"
      "            prev_injection,\n")
b3 = ("        if _FN_HCFUSE and self.use_combine and prev_injection is not None:  " + M + "\n"
      "            hidden_states, block_input, d = hc_combine_mix_fused(  " + M + "\n"
      "                hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,  " + M + "\n"
      "                self.input_mix_weight_down_block_inject.weight, self.input_mix_weight_up.weight,  " + M + "\n"
      "                self.config.rms_norm_eps, self.hc_count, self.lora_rank)  " + M + "\n"
      "            return hidden_states, block_input, d[:, self.lora_rank:self.lora_rank + self.hc_count]  " + M + "\n"
      + a3)
for a, b in ((a1, b1), (a2, b2), (a3, b3)):
    assert s.count(a) == 1, (a[:40], s.count(a))
    s = s.replace(a, b)
compile(s, HCF, "exec")
shutil.copy2(SRC, OPS)
open(HCF, "w").write(s)
print("hcfuse on", s.count(M))
