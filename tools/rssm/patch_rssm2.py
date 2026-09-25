"""FNRSSM phase 2 on top of patch_rssm.py: PLE short conv on the RecoverSSM protocol + align mode. argv[1] = vllm pkg dir."""
import sys, shutil, os
V = sys.argv[1]; MARK = "FNRSSM2"
def patch(rel, edits):
    p = os.path.join(V, rel); s = open(p).read()
    if MARK in s:
        print("already patched:", rel); return
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched:", rel)

patch("model_executor/layers/mamba/abstract.py", [(
"""                # FNRSSM: short-conv (Qwen4Exp PLE) layers keep the native per-draft protocol
                and getattr(getattr(self, "mamba_type", None), "name", "") != "SHORT_CONV"
""",
"""                # FNRSSM2: the PLE short conv follows the RecoverSSM protocol too (one block, compacted window)
""")])
patch("config/vllm.py", [(
"""            if self.cache_config.mamba_cache_mode != "none":
                raise ValueError(
                    "FN_GDN_RECOVERSSM: only mamba_cache_mode='none' so far (align needs the PLE short-conv "
                    "commit); run with --no-enable-prefix-caching"
                )""",
"""            if self.cache_config.mamba_cache_mode not in ("none", "align"):
                raise ValueError("FN_GDN_RECOVERSSM supports mamba_cache_mode none and align")
            if self.cache_config.mamba_cache_mode == "align" and not self.use_v2_model_runner:
                raise ValueError("FN_GDN_RECOVERSSM with align mode requires the V2 model runner")
            # FNRSSM2: align mode supported (PLE short conv on the RecoverSSM protocol)""")])
patch("models/qwen4_exp/nvidia/ple_layer.py", [(
"""    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend
""",
"""    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        if getattr(self.cache_config, "use_kda_recoverssm", False) and self.num_spec_tokens > 0:
            from vllm.v1.attention.backends.ple_recoverssm import PleRecoverSSMAttentionBackend
            return PleRecoverSSMAttentionBackend  # FNRSSM2
        return PleShortConvAttentionBackend
""")])
shutil.copy("/opt/llm/runners/rssm/ple_recoverssm.py", os.path.join(V, "v1/attention/backends/ple_recoverssm.py"))
print("copied ple_recoverssm.py")
