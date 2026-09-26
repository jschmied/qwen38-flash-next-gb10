"""FNDYN2 — FNDYN (confidence-gated early exit of MTP drafting) plus the confidence on the FNNVFP4 draft-head branch
(jschmied 2026-09-26). argv[1] = vllm package dir, argv[2] == "off" removes. Original FNDYN notes follow.
FNDYN — confidence-gated early exit of MTP drafting (jschmied 2026-09-24, local experiment).
Inert unless FN_DYN_DRAFT=1; threshold from FN_DYN_DRAFT_THR_FILE (re-read on change) or FN_DYN_DRAFT_THR; 0 = off. Two files in the main venv:
  mtp.py          get_top_tokens (BF16 draft-vocab slice path) also stores the draft's max softmax prob
  speculator.py   _multi_step_decode stops drafting once every request's running confidence product < threshold;
                  the unused draft slots repeat the last draft token (the target verifies them like any other)
`off` removes both byte-exactly. Optional argv[2]: a directory holding mtp.py + speculator.py copies (dry run)."""
import os, sys
VENV = sys.argv[1]
MTP = f"{VENV}/models/qwen4_exp/nvidia/mtp.py"
SPEC = f"{VENV}/v1/worker/gpu/spec_decode/autoregressive/speculator.py"

M1 = "_fn_logger = _fn_init_logger(__name__)\n"
M1N = M1 + '_FN_DYN = _fn_os.environ.get("FN_DYN_DRAFT", "") == "1"  # FNDYN\n'
M2 = """        logits = torch.nn.functional.linear(hidden_states.to(w.dtype), w)
        return self._fn_draft_ids[logits.argmax(dim=-1)]
"""
M2N = """        logits = torch.nn.functional.linear(hidden_states.to(w.dtype), w)
        if _FN_DYN:  # FNDYN: draft confidence = max softmax prob over the slice
            _lf = logits.float()  # FNDYN
            self._fn_last_conf = torch.exp(_lf.max(dim=-1).values - torch.logsumexp(_lf, dim=-1))  # FNDYN
        return self._fn_draft_ids[logits.argmax(dim=-1)]
"""

S1 = "logger = init_logger(__name__)\n"
S1N = S1 + """
# ---- FNDYN head (jschmied 2026-09-24) ----
import os as _fn_os  # FNDYN
_FN_DYN = _fn_os.environ.get("FN_DYN_DRAFT", "") == "1"  # FNDYN
_FN_DYN_FILE = _fn_os.environ.get("FN_DYN_DRAFT_THR_FILE", "")  # FNDYN
_FN_DYN_STATE = {"thr": float(_fn_os.environ.get("FN_DYN_DRAFT_THR", "0") or 0), "mtime": None}  # FNDYN
_FN_DYN_STATS = {"cycles": 0}  # FNDYN
if _FN_DYN:  # FNDYN
    logger.warning("FNDYN active: threshold %.3f, file %s", _FN_DYN_STATE["thr"], _FN_DYN_FILE or "-")


def _fn_dyn_thr() -> float:  # FNDYN: re-read the threshold file when it changes (one stat per cycle)
    if _FN_DYN_FILE:
        try:
            m = _fn_os.stat(_FN_DYN_FILE).st_mtime_ns
            if m != _FN_DYN_STATE["mtime"]:
                _FN_DYN_STATE["mtime"] = m
                _FN_DYN_STATE["thr"] = float(open(_FN_DYN_FILE).read().strip() or 0)
                logger.info("FNDYN threshold -> %.3f", _FN_DYN_STATE["thr"])
        except (OSError, ValueError):
            pass
    return _FN_DYN_STATE["thr"]


def _fn_dyn_count(key) -> None:  # FNDYN
    _FN_DYN_STATS[key] = _FN_DYN_STATS.get(key, 0) + 1
    _FN_DYN_STATS["cycles"] += 1
    if _FN_DYN_STATS["cycles"] % 500 == 0:
        logger.info("FNDYN stops after %d cycles: %s", _FN_DYN_STATS["cycles"],
                    {k: v for k, v in sorted(_FN_DYN_STATS.items()) if k != "cycles"})
# ---- end FNDYN head ----
"""
S2 = """        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
"""
S2N = """        attn_metadata = None
        slot_mappings_by_layer = None
        _fn_cum = None  # FNDYN
        _fn_stopped = False  # FNDYN
        _fn_thr = _fn_dyn_thr() if _FN_DYN else 0.0  # FNDYN
        for step in range(1, self.num_speculative_steps):
            if _fn_thr > 0 and not skip_attn and batch_desc.cg_mode != CUDAGraphMode.FULL:  # FNDYN
                _c = getattr(self.model, "_fn_last_conf", None)  # FNDYN
                if _c is not None and _c.shape[0] >= num_reqs:  # FNDYN
                    _c = _c[:num_reqs]  # FNDYN
                    _fn_cum = _c.clone() if _fn_cum is None else _fn_cum * _c  # FNDYN
                    if float(_fn_cum.max()) < _fn_thr:  # FNDYN (one host sync per draft step)
                        self.draft_tokens[:num_reqs, step:] = self.draft_tokens[:num_reqs, step - 1 : step]  # FNDYN
                        _fn_dyn_count(f"stop_before_step{step}")  # FNDYN
                        _fn_stopped = True  # FNDYN
                        break  # FNDYN
            # Rebuild every step when positions advance, or just once
"""
S3 = """                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def _fused_multi_step_decode(
"""
S3N = """                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )
        if _fn_thr > 0 and not _fn_stopped and _fn_cum is not None:  # FNDYN
            _fn_dyn_count("full")  # FNDYN

    def _fused_multi_step_decode(
"""
M3 = """            _lg = nvfp4_rows_gemv(hidden_states.to(torch.bfloat16), _q4, self._fn_draft_s, self._fn_draft_g)  # FNNVFP4
            return self._fn_draft_ids[_lg.argmax(dim=-1)]  # FNNVFP4
"""
M3N = """            _lg = nvfp4_rows_gemv(hidden_states.to(torch.bfloat16), _q4, self._fn_draft_s, self._fn_draft_g)  # FNNVFP4
            if _FN_DYN:  # FNDYN: the same confidence on the NVFP4 head's float32 logits
                self._fn_last_conf = torch.exp(_lg.max(dim=-1).values - torch.logsumexp(_lg, dim=-1))  # FNDYN
            return self._fn_draft_ids[_lg.argmax(dim=-1)]  # FNNVFP4
"""
EDITS = {MTP: [(M1, M1N), (M2, M2N), (M3, M3N)], SPEC: [(S1, S1N), (S2, S2N), (S3, S3N)]}
off = sys.argv[2:3] == ["off"]
for path, pairs in EDITS.items():
    s = open(path).read()
    if off:
        if "FNDYN" not in s:
            print(f"  {os.path.basename(path)}: not installed"); continue
        for a, n in pairs:
            s = s.replace(n, a)
        assert "FNDYN" not in s, f"partial removal in {path}"
        open(path, "w").write(s); print(f"  {os.path.basename(path)}: FNDYN REMOVED")
    else:
        if "FNDYN" in s:
            print(f"  {os.path.basename(path)}: already installed"); continue
        for a, n in pairs:
            assert s.count(a) == 1, f"anchor not unique/absent in {path}: {a[:50]!r}"
        for a, n in pairs:
            s = s.replace(a, n)
        open(path, "w").write(s); print(f"  {os.path.basename(path)}: FNDYN INSTALLED (inert unless FN_DYN_DRAFT=1)")
