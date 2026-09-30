"""Install/remove FNHEAD4 (NVFP4 W4A16 target lm_head) in a venv. Inert unless FN_TARGET_HEAD_NVFP4=1.
  python patch_head4.py <site-packages>/vllm [off] [model.py-copy-for-dry-run]
on: copy fn_head4.py next to model.py, hook Qwen4Exp compute_logits (one anchor). off: byte-exact removal."""
import os, shutil, sys
VLLM = sys.argv[1]
OFF = sys.argv[2:3] == ["off"]
PKG = os.path.join(VLLM, "models/qwen4_exp/nvidia")
MODEL = sys.argv[3] if len(sys.argv) > 3 else os.path.join(PKG, "model.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_head4.py")
DST = os.path.join(os.path.dirname(MODEL), "fn_head4.py")

A = """    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
"""
N = """    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        import os as _fn_h4_os  # FNHEAD4
        if _fn_h4_os.environ.get("FN_TARGET_HEAD_NVFP4", "") == "1":  # FNHEAD4
            from vllm.models.qwen4_exp.nvidia.fn_head4 import target_logits  # FNHEAD4
            _lg = target_logits(self, hidden_states)  # FNHEAD4
            if _lg is not None:  # FNHEAD4
                return _lg  # FNHEAD4
        return self.logits_processor(self.lm_head, hidden_states)
"""
s = open(MODEL).read()
if OFF:
    if "FNHEAD4" in s:
        s = s.replace(N, A)
        assert "FNHEAD4" not in s, "partial removal"
        open(MODEL, "w").write(s); print("  fnhead4 REMOVED")
    else:
        print("  fnhead4 not installed")
    if os.path.exists(DST):
        os.remove(DST)
else:
    if "FNHEAD4" in s:
        print("  fnhead4 already installed")
    else:
        assert s.count(A) == 1, "anchor"
        open(MODEL, "w").write(s.replace(A, N)); print("  fnhead4 INSTALLED (inert unless FN_TARGET_HEAD_NVFP4=1)")
    shutil.copyfile(SRC, DST)
