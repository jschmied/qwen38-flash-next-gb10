"""Install/remove the NVFP4 draft-head slice (FNNVFP4) in the main venv. Inert unless FN_DRAFT_HEAD_NVFP4=1.
  on : copy fn_nvfp4_head.py into the package, patch mtp.py (two anchors)
  off: byte-exact removal of both.   Optional argv[2]: target mtp.py (for dry runs on a copy)."""
import os, shutil, sys
PKG = "/opt/llm/runtime/vllm-venv-main1ea7/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia"
MTP = sys.argv[2] if len(sys.argv) > 2 else f"{PKG}/mtp.py"
MOD_DST = os.path.join(os.path.dirname(MTP), "fn_nvfp4_head.py")
MOD_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_nvfp4_head.py")

A1 = """        deq = (rows * srows.repeat_interleave(bk, dim=1)).to(torch.bfloat16).contiguous()
    model.register_buffer("_fn_draft_weight", deq, persistent=False)
"""
N1 = """        wf = rows * srows.repeat_interleave(bk, dim=1)  # FNNVFP4
        deq = wf.to(torch.bfloat16).contiguous()
    # ---- FNNVFP4 (jschmied 2026-09-24): FN_DRAFT_HEAD_NVFP4=1 keeps the slice as NVFP4, not BF16 ----
    if _fn_os.environ.get("FN_DRAFT_HEAD_NVFP4", "") == "1":
        from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import quantize_nvfp4_rows
        with torch.no_grad():
            q4, s4, g4 = quantize_nvfp4_rows(wf)
        del wf, deq
        model.register_buffer("_fn_draft_q", q4, persistent=False)
        model.register_buffer("_fn_draft_s", s4, persistent=False)
        model._fn_draft_g = g4
        model.register_buffer("_fn_draft_ids", index.to(torch.int64), persistent=False)
        _mib = lambda t: t.numel() * t.element_size() / 2**20
        _fn_logger.info("FNDV draft vocab: %d of %d rows (%.1f%%); draft head %.0f -> %.0f MiB per draft step "
                        "(NVFP4 slice, FNNVFP4)", len(ids), N, 100.0 * len(ids) / N, _mib(w) + _mib(s),
                        _mib(q4) + _mib(s4))
        return
    # ---- end FNNVFP4 ----
    model.register_buffer("_fn_draft_weight", deq, persistent=False)
"""
A2 = """        w = getattr(self, "_fn_draft_weight", None)
        if w is None:
"""
N2 = """        _q4 = getattr(self, "_fn_draft_q", None)  # FNNVFP4
        if _q4 is not None:  # FNNVFP4
            from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import nvfp4_rows_gemv  # FNNVFP4
            _lg = nvfp4_rows_gemv(hidden_states.to(torch.bfloat16), _q4, self._fn_draft_s, self._fn_draft_g)  # FNNVFP4
            return self._fn_draft_ids[_lg.argmax(dim=-1)]  # FNNVFP4
        w = getattr(self, "_fn_draft_weight", None)
        if w is None:
"""
s = open(MTP).read()
if sys.argv[1:2] == ["off"]:
    if "FNNVFP4" not in s:
        print("  fnnvfp4 not installed")
    else:
        s = s.replace(N1, A1).replace(N2, A2)
        assert "FNNVFP4" not in s, "partial removal"
        open(MTP, "w").write(s); print("  fnnvfp4 REMOVED")
    if os.path.exists(MOD_DST):
        os.remove(MOD_DST)
else:
    if "FNNVFP4" in s:
        print("  fnnvfp4 already installed")
    else:
        assert s.count(A1) == 1 and s.count(A2) == 1, "anchor"
        open(MTP, "w").write(s.replace(A1, N1).replace(A2, N2)); print("  fnnvfp4 INSTALLED (inert unless FN_DRAFT_HEAD_NVFP4=1)")
    shutil.copyfile(MOD_SRC, MOD_DST)
