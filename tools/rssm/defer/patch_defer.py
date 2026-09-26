"""FNRSSM deferred commit (on top of patch_rssm/sticky/unpack/rssm2): copies the deferred-capable recoverssm_gdn.py,
sizes the replay record by replay_record_dim(), passes the pending counters into the verify, and clears them for
prefill rows in the builder. Gate: FN_GDN_RECOVERSSM_DEFER=1 (with FN_GDN_RECOVERSSM=1); off, everything behaves as
before. argv[1] = vllm package dir; argv[2] == "off" removes the edits (restore recoverssm_gdn.py from its backup)."""
import os, shutil, sys

V = sys.argv[1]; OFF = len(sys.argv) > 2 and sys.argv[2] == "off"; MARK = "FNRSSMDEFER"
HERE = os.path.dirname(os.path.abspath(__file__))
EDITS = {
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": [
        ("""            shapes = (*shapes, (self.num_v_heads // self.tp_size, self.num_spec + 1,
                                self.head_v_dim + self.head_k_dim + 1))""",
         """            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import replay_record_dim  # FNRSSMDEFER
            shapes = (*shapes, (self.num_v_heads // self.tp_size, self.num_spec + 1,
                                replay_record_dim(self.head_k_dim, self.head_v_dim)))"""),
        ("""                spec_query_len=self.num_spec + 1, use_qk_l2norm_in_kernel=True)""",
         """                spec_query_len=self.num_spec + 1, use_qk_l2norm_in_kernel=True,
                pending=getattr(getattr(attn_metadata, "recoverssm_context", None), "pending", None))  # FNRSSMDEFER"""),
    ],
    "models/qwen4_exp/nvidia/model.py": [
        ("""            shapes = (*shapes, (hf_config.linear_num_value_heads // tp_size, num_spec + 1,
                                hf_config.linear_value_head_dim + hf_config.linear_key_head_dim + 1))""",
         """            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import replay_record_dim  # FNRSSMDEFER
            shapes = (*shapes, (hf_config.linear_num_value_heads // tp_size, num_spec + 1,
                                replay_record_dim(hf_config.linear_key_head_dim, hf_config.linear_value_head_dim)))"""),
    ],
    "v1/attention/backends/gdn_recoverssm.py": [
        ("""        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        commit = None""",
         """        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        n_non_spec = meta.num_prefills + meta.num_decodes
        if n_non_spec and meta.non_spec_state_indices_tensor is not None:  # FNRSSMDEFER
            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import DEFERRED
            if DEFERRED:
                # A block starting a prefill may carry a previous owner's pending records: forget them.
                self._get_context().clear_pending(meta.non_spec_state_indices_tensor[:n_non_spec])
        commit = None"""),
    ],
}
for rel, edits in EDITS.items():
    p = os.path.join(V, rel); s = open(p).read()
    if OFF:
        for old, new in edits:
            s = s.replace(new, old)
        assert MARK not in s, rel
    else:
        if MARK in s:
            print("already:", rel); continue
        for old, new in edits:
            assert s.count(old) == 1, (rel, old[:70], s.count(old))
            s = s.replace(old, new)
    open(p, "w").write(s); print(("removed: " if OFF else "patched: ") + rel)
dst = os.path.join(V, "model_executor/layers/mamba/gdn/recoverssm_gdn.py")
if OFF:
    shutil.copy(dst + ".orig-defer", dst); print("restored recoverssm_gdn.py")
else:
    if not os.path.exists(dst + ".orig-defer"):
        shutil.copy(dst, dst + ".orig-defer")
    shutil.copy(os.path.join(HERE, "recoverssm_gdn.py"), dst); print("copied deferred recoverssm_gdn.py")
