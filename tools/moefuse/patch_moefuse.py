"""FNMOEFUSE overlay: copy moe_fp4.py as fused_moe/fn_moe_fp4.py and hook FlashInferExperts.apply (env FN_MOEFUSE=1,
prefill rows >= FN_MOEFUSE_MIN, default 128). Usage: python patch_moefuse.py <site-packages>/vllm [off].
On: backs up flashinfer_cutlass_moe.py once (*.orig-moefuse). Off: restores it and removes fn_moe_fp4.py.
VLLM_X_PY=<file> patches that file instead (dry run on a copy)."""
import os, shutil, sys
pkg = sys.argv[1]; off = len(sys.argv) > 2 and sys.argv[2] == "off"
D = os.path.join(pkg, "model_executor/layers/fused_moe")
F = os.environ.get("VLLM_X_PY") or os.path.join(D, "experts/flashinfer_cutlass_moe.py")
OPS = os.path.join(D, "fn_moe_fp4.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_fp4.py")
M = "# FNMOEFUSE"
if off:
    shutil.copy2(F + ".orig-moefuse", F)
    if not os.environ.get("VLLM_X_PY") and os.path.exists(OPS):
        os.remove(OPS)
    print("moefuse off", M in open(F).read()); sys.exit(0)
s = open(F).read()
if M in s:
    print("already on"); sys.exit(0)
if not os.path.exists(F + ".orig-moefuse"):
    shutil.copy2(F, F + ".orig-moefuse")
a1 = "def is_valid_flashinfer_cutlass_fused_moe(\n"
b1 = ("import os as _fn_os  " + M + "\n"
      "_FN_MOEFUSE = _fn_os.environ.get(\"FN_MOEFUSE\") == \"1\"  " + M + "\n"
      "_FN_MOEFUSE_MIN = int(_fn_os.environ.get(\"FN_MOEFUSE_MIN\", \"128\"))  " + M + "\n"
      "if _FN_MOEFUSE:  " + M + "\n"
      "    from vllm.model_executor.layers.fused_moe.fn_moe_fp4 import moe_fp4_prefill as _fn_moe_fp4  " + M + "\n"
      "\n\n" + a1)
a2 = ("        quant_scales = None\n"
      "        fc1_expert_weights = None\n")
b2 = ("        if (_FN_MOEFUSE and self.quant_dtype == \"nvfp4\" and activation == MoEActivation.SILU  " + M + "\n"
      "                and hidden_states.shape[0] >= _FN_MOEFUSE_MIN and hidden_states.dtype == torch.uint8  " + M + "\n"
      "                and a1q_scale is not None and expert_map is None and not apply_router_weight_on_input  " + M + "\n"
      "                and self.gemm1_clamp_limit is None and getattr(self, \"w1_bias\", None) is None  " + M + "\n"
      "                and self.tp_size == 1 and self.ep_size == 1 and self.out_dtype == torch.bfloat16  " + M + "\n"
      "                and _fn_os.environ.get(\"VLLM_MOE_DET_FINALIZE\")):  " + M + "\n"
      "            _fn_moe_fp4(hidden_states, a1q_scale, w1, self.w1_scale, w2, self.w2_scale, self.g1_alphas,  " + M + "\n"
      "                        self.a2_gscale, self.g2_alphas, topk_ids, topk_weights, output)  " + M + "\n"
      "            return  " + M + "\n"
      + a2)
for a, b in ((a1, b1), (a2, b2)):
    assert s.count(a) == 1, (a[:40], s.count(a))
    s = s.replace(a, b)
compile(s, F, "exec")
if not os.environ.get("VLLM_X_PY"):
    shutil.copy2(SRC, OPS)
open(F, "w").write(s)
print("moefuse on", s.count(M))
