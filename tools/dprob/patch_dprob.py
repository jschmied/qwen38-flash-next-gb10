"""FNDRAFTPROB: probabilistic MTP drafting over the 32k NVFP4 draft-vocab slice.
With FN_DRAFT_PROB=1 (and draft_sample_method='probabilistic', no local argmax), Qwen4ExpMTP.compute_logits returns
the slice logits scattered into a persistent full-vocab buffer filled with -inf once: the speculator's gumbel sampler
and the rejection sampler then use exactly the drafter's sampling distribution (q = 0 outside the slice, so the ratio
test stays exact). Usage: python patch_dprob.py [on|off]; target via VLLM_X_PY (default: clone venv mtp.py)."""
import os, sys
T = os.environ.get("VLLM_X_PY", "/opt/llm/runtime/vllm-venv-rssm/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/mtp.py")
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
s = open(T).read()
OLD = '''    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
'''
NEW = '''    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        if _fn_os.environ.get("FN_DRAFT_PROB", "") == "1":  # FNDRAFTPROB
            if getattr(self, "_fn_draft_q", None) is None and not getattr(self, "_fn_prob_tried", False):  # FNDRAFTPROB
                self._fn_prob_tried = True  # FNDRAFTPROB
                self._fn_attach_draft_vocab()  # FNDRAFTPROB
            _q4 = getattr(self, "_fn_draft_q", None)  # FNDRAFTPROB
            if _q4 is not None:  # FNDRAFTPROB
                from vllm.models.qwen4_exp.nvidia.fn_nvfp4_head import nvfp4_rows_gemv  # FNDRAFTPROB
                _lg = nvfp4_rows_gemv(hidden_states.to(torch.bfloat16), _q4, self._fn_draft_s, self._fn_draft_g)  # FNDRAFTPROB
                _m = _lg.shape[0]  # FNDRAFTPROB
                _buf = getattr(self, "_fn_prob_buf", None)  # FNDRAFTPROB
                if _buf is None or _buf.shape[0] < _m:  # FNDRAFTPROB
                    _n = self.lm_head.weight.shape[0] if self.lm_head.weight.dim() == 2 else self.config.vocab_size  # FNDRAFTPROB
                    _buf = torch.full((max(_m, 64), _n), float("-inf"), dtype=torch.float32,  # FNDRAFTPROB
                                      device=hidden_states.device)  # FNDRAFTPROB
                    self._fn_prob_buf = _buf  # FNDRAFTPROB
                    _fn_logger.warning("FNDRAFTPROB path taken: probabilistic drafting over a %d-id slice of %d",  # FNDRAFTPROB
                                       self._fn_draft_ids.numel(), _n)  # FNDRAFTPROB
                _out = _buf[:_m]  # FNDRAFTPROB
                _out[:, self._fn_draft_ids] = _lg  # FNDRAFTPROB
                return _out  # FNDRAFTPROB
        return self.logits_processor(self.lm_head, hidden_states)
'''
if MODE == "on":
    if "FNDRAFTPROB" in s: sys.exit("already on")
    if s.count(OLD) != 1: sys.exit(f"anchor count {s.count(OLD)}")
    s = s.replace(OLD, NEW)
else:
    if s.count(NEW) != 1: sys.exit("remove anchor missing")
    s = s.replace(NEW, OLD)
open(T, "w").write(s); print("FNDRAFTPROB", MODE, T)
