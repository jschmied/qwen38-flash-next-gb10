#!/usr/bin/env python3
"""1ea7 port of blazux/qwen3.8-Flash-DGX src/patch_mtp_name_prefilter.py (patch 18, Apache-2.0): the MTP drafter
skips checkpoint tensors its name remap would drop, before they are read. VLLM_MTP_NAME_PREFILTER=0 disables."""
import ast
import os
import re
import sys

SP = sys.argv[1]
WU = f"{SP}/vllm/model_executor/model_loader/weight_utils.py"
MTP = f"{SP}/vllm/models/qwen4_exp/nvidia/mtp.py"
assert os.path.exists(MTP), f"mtp.py not found: {MTP}"

s = open(WU).read()
assert "from vllm.model_executor.model_loader.ep_weight_filter import (\n    should_skip_weight,\n)" in s
assert s.count("should_skip_weight(name, local_expert_ids)") >= 2
s += '''

# --- qwen38-flash-dgx: optional keep-filter in front of should_skip_weight (patch 18) ---
import contextlib as _qwen38_pf_contextlib

_qwen38_keep_name = None
_qwen38_keep_stats = [0, 0]  # kept, skipped
_qwen38_should_skip_weight = should_skip_weight


def should_skip_weight(weight_name, local_expert_ids):  # noqa: F811
    keep = _qwen38_keep_name
    if keep is not None:
        if not keep(weight_name):
            _qwen38_keep_stats[1] += 1
            return True
        _qwen38_keep_stats[0] += 1
    return _qwen38_should_skip_weight(weight_name, local_expert_ids)


@_qwen38_pf_contextlib.contextmanager
def qwen38_keep_only(keep, what: str):
    global _qwen38_keep_name
    prev, _qwen38_keep_name = _qwen38_keep_name, keep
    _qwen38_keep_stats[:] = [0, 0]
    try:
        yield
    finally:
        _qwen38_keep_name = prev
        logger.info("qwen38 %s name prefilter: %d tensors kept, %d skipped before reading "
                    "(VLLM_MTP_NAME_PREFILTER=0 disables)", what, *_qwen38_keep_stats)
'''
ast.parse(s)
open(WU, "w").write(s)

# ---- 1ea7 port of the mtp.py half (jschmied 2026-09-28): our mtp.py wraps the loader call in the SCALEINV-MTP
# overlay, as `return loader.load_weights(...)` (prod) or `_fn_loaded = loader.load_weights(...)` (clone venv).
m = open(MTP).read()
ret = re.compile(
    r"^        ((?:return |_fn_loaded = )loader\.load_weights\(_scaleinv\(remap_weight_names\(\)\), mapper=mapper\).*)\n",
    re.M,
)
found = ret.findall(m)
assert len(found) == 1, f"MTP load_weights call: expected 1, found {len(found)}"


def _wrap(g):
    return (
        "        from vllm.model_executor.model_loader import weight_utils as _q38_wu  # PF18\n"
        "        _q38_ctx = (  # PF18\n"
        "            _qwen38_ctx_pf.nullcontext()  # PF18\n"
        "            if _qwen38_os_pf.environ.get(\"VLLM_MTP_NAME_PREFILTER\", \"1\") == \"0\"  # PF18\n"
        "            or getattr(self, \"secondary_weights\", ())  # PF18\n"
        "            else _q38_wu.qwen38_keep_only(  # PF18\n"
        "                lambda n: _remap_mtp_weight_name(n) is not None, \"MTP\"  # PF18\n"
        "            )  # PF18\n"
        "        )  # PF18\n"
        "        with _q38_ctx:  # PF18\n"
        "            " + g.group(1) + "\n"
    )


m = ret.sub(_wrap, m)
anchor = "def _remap_mtp_weight_name(name: str) -> str | None:\n"
assert m.count(anchor) == 1, "_remap_mtp_weight_name not found"
m = m.replace(anchor, "import contextlib as _qwen38_ctx_pf  # PF18\nimport os as _qwen38_os_pf  # PF18\n\n\n" + anchor)
ast.parse(m)
open(MTP, "w").write(m)
print("weight_utils.py + mtp.py: MTP name prefilter (1ea7 port) applied OK")
