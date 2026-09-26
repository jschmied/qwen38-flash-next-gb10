"""A1 (speed-of-light 5f): under RecoverSSM spec verify, give gdn_recoverssm_verify strided VIEWS of the conv output
for q/k/v (no cat + 3 copies) and let it write straight into core_attn_out (no 49 KB D2D copy). Same kernel, same
values. Env-gated: FNQKVVIEW=1. Usage: python patch_qkvview.py [on|off]; target path via VLLM_X_PY or the venv."""
import os, sys
T = os.environ.get("VLLM_X_PY", "/opt/llm/runtime/vllm-venv-rssm/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
s = open(T).read()
A_OLD = "        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)\n"
A_NEW = """        _fnqv = (_FNQKVVIEW and mixed_qkv_spec is not None and spec_sequence_masks is not None
                 and getattr(self, "use_gdn_recoverssm", False))  # FNQKVVIEW
        if _fnqv:  # FNQKVVIEW
            _qd = self.key_dim // self.tp_size; _vd = self.value_dim // self.tp_size; _T = mixed_qkv_spec.shape[0]
            query_spec = mixed_qkv_spec[:, :_qd].view(1, _T, -1, self.head_k_dim)
            key_spec = mixed_qkv_spec[:, _qd:2 * _qd].view(1, _T, -1, self.head_k_dim)
            value_spec = mixed_qkv_spec[:, 2 * _qd:2 * _qd + _vd].view(1, _T, -1, self.head_v_dim)
            global _FNQKVVIEW_LOGGED
            if not _FNQKVVIEW_LOGGED:
                _FNQKVVIEW_LOGGED = True
                logger.warning("FNQKVVIEW path taken: strided q/k/v views, token stride %d", query_spec.stride(1))
        else:  # FNQKVVIEW
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)  # FNQKVVIEW
"""
B_OLD = """                spec_query_len=self.num_spec + 1, use_qk_l2norm_in_kernel=True)
            last_recurrent_state = None
"""
B_NEW = """                spec_query_len=self.num_spec + 1, use_qk_l2norm_in_kernel=True,
                out=(core_attn_out[:num_actual_tokens].unsqueeze(0)  # FNQKVVIEW
                     if _fnqv and mixed_qkv_non_spec is None and query_spec.shape[1] == num_actual_tokens else None))
            last_recurrent_state = None
"""
C_OLD = """        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
"""
C_NEW = """        elif spec_sequence_masks is not None:
            if core_attn_out_spec.data_ptr() != core_attn_out.data_ptr():  # FNQKVVIEW: skip when verify wrote in place
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
"""
D_OLD = "logger = init_logger(__name__)\n"
D_NEW = D_OLD + '_FNQKVVIEW = os.environ.get("FNQKVVIEW", "") == "1"  # FNQKVVIEW\n_FNQKVVIEW_LOGGED = False  # FNQKVVIEW\n'
pairs = [(A_OLD, A_NEW), (B_OLD, B_NEW), (C_OLD, C_NEW), (D_OLD, D_NEW)]
if MODE == "on":
    if "FNQKVVIEW" in s: sys.exit("already on")
    for o, n in pairs:
        if s.count(o) != 1: sys.exit(f"anchor count {s.count(o)}: {o[:60]!r}")
        s = s.replace(o, n)
    if "\nimport os\n" not in s: s = "import os  # FNQKVVIEW\n" + s
else:
    if "FNQKVVIEW" not in s: sys.exit("already off")
    for o, n in pairs:
        if s.count(n) != 1: sys.exit(f"remove anchor count {s.count(n)}")
        s = s.replace(n, o)
    s = s.replace("import os  # FNQKVVIEW\n", "")
    if "FNQKVVIEW" in s: sys.exit("leftover marker")
open(T, "w").write(s); print("FNQKVVIEW", MODE, T)
