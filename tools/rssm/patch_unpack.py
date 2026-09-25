import sys, os
p = os.path.join(sys.argv[1], "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"); s = open(p).read()
n = 0
for old, new in (("        _, state_dtype = self.get_state_dtype()\n", "        state_dtype = self.get_state_dtype()[1]  # FNRSSM: 3 dtypes with the replay record\n"),
                 ("        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()\n",
                  "        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()[:2]  # FNRSSM\n")):
    c = s.count(old); n += c; s = s.replace(old, new)
open(p, "w").write(s); print("replaced", n)
