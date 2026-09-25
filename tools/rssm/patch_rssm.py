"""Install the GDN RecoverSSM overlay into a venv's vllm package (argv[1] = .../site-packages/vllm)."""
import sys, shutil, os
V = sys.argv[1]; MARK = "FNRSSM"
def patch(rel, edits):
    p = os.path.join(V, rel); s = open(p).read()
    if MARK in s:
        print("already patched:", rel); return
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:80], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched:", rel)

# 1. zero per-draft state blocks for RecoverSSM, but not for short-conv (PLE) layers
patch("model_executor/layers/mamba/abstract.py", [(
"""                0
                if vllm_config.cache_config.use_kda_recoverssm
                else vllm_config.num_speculative_tokens""",
"""                0
                if vllm_config.cache_config.use_kda_recoverssm
                # FNRSSM: short-conv (Qwen4Exp PLE) layers keep the native per-draft protocol
                and getattr(getattr(self, "mamba_type", None), "name", "") != "SHORT_CONV"
                else vllm_config.num_speculative_tokens""")])

# 2. config: FN_GDN_RECOVERSSM=1 turns RecoverSSM on for Qwen4Exp GDN layers
patch("config/vllm.py", [(
"""    def validate_mamba_cached_kernel(self) -> "VllmConfig":
        if not self.cache_config.use_replayssm:""",
"""    def validate_mamba_cached_kernel(self) -> "VllmConfig":
        # FNRSSM (jschmied 2026-09-25, local): GDN RecoverSSM for Qwen4Exp, reusing the KDA RecoverSSM plumbing.
        if (
            os.environ.get("FN_GDN_RECOVERSSM", "") == "1"
            and self.num_speculative_tokens > 0
            and self.model_config is not None
            and self.model_config.architecture == "Qwen4ExpForConditionalGeneration"
        ):
            if self.cache_config.mamba_cache_mode != "none":
                raise ValueError(
                    "FN_GDN_RECOVERSSM: only mamba_cache_mode='none' so far (align needs the PLE short-conv "
                    "commit); run with --no-enable-prefix-caching"
                )
            self.cache_config.use_kda_recoverssm = True
            return self
        if not self.cache_config.use_replayssm:""")])

# 3. Qwen GDN layer
patch("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", [
("""        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
""",
"""        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
        # FNRSSM: RecoverSSM verify replaces the per-draft snapshot path (Triton decode only)
        self.use_gdn_recoverssm = bool(getattr(self.cache_config, "use_kda_recoverssm", False)) and self.num_spec > 0
        if self.use_gdn_recoverssm:
            self.gdn_decode_kernel = "triton"
            self.enable_fused_gdn_decode = False
            self.enable_packed_recurrent_decode = False
            logger.warning_once("FNRSSM: GDN RecoverSSM speculative verify active (spec_query_len %d)",
                                self.num_spec + 1)
"""),
("""    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )
""",
"""    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )
        if getattr(self.cache_config, "use_kda_recoverssm", False) and self.num_spec > 0:
            # FNRSSM: per-token replay record [HV, spec_query_len, V + K + 1] fp32
            shapes = (*shapes, (self.num_v_heads // self.tp_size, self.num_spec + 1,
                                self.head_v_dim + self.head_k_dim + 1))
        return shapes

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = super().get_state_dtype()
        if getattr(self.cache_config, "use_kda_recoverssm", False) and self.num_spec > 0:
            dtypes = (*dtypes, torch.float32)  # FNRSSM
        return dtypes

    def get_attn_backend(self):
        if getattr(self.cache_config, "use_kda_recoverssm", False) and self.num_spec > 0:
            from vllm.v1.attention.backends.gdn_recoverssm import GDNRecoverSSMAttentionBackend
            return GDNRecoverSSMAttentionBackend  # FNRSSM
        return super().get_attn_backend()
"""),
("""                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )""",
"""                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=(self.num_spec + 1 if getattr(self, "use_gdn_recoverssm", False)  # FNRSSM
                               else spec_state_indices_tensor.size(-1)),
                validate_data=False,
            )"""),
("""        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(""",
"""        if spec_sequence_masks is not None and getattr(self, "use_gdn_recoverssm", False):
            # FNRSSM: verify the window off the checkpoint; the commit after sampling writes the state once
            from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import gdn_recoverssm_verify
            _n = attn_metadata.num_spec_decodes
            core_attn_out_spec = gdn_recoverssm_verify(
                self.A_log, a_spec, b_spec, self.dt_bias, query_spec, key_spec, value_spec,
                checkpoint_state=ssm_state, replay_cache=self_kv_cache[2],
                query_start_loc=spec_query_start_loc[: _n + 1],
                state_indices=spec_state_indices_tensor[:_n, 0],
                spec_query_len=self.num_spec + 1, use_qk_l2norm_in_kernel=True)
            last_recurrent_state = None
        elif spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update("""),
])

# 4. Qwen4Exp cache planning: the GDN spec must include the replay record
patch("models/qwen4_exp/nvidia/model.py", [
("""        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
""",
"""        dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        if vllm_config.cache_config.use_kda_recoverssm:  # FNRSSM
            dtypes = (*dtypes, torch.float32)
        return dtypes
"""),
("""        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )
""",
"""        shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )
        if vllm_config.cache_config.use_kda_recoverssm and num_spec > 0:  # FNRSSM
            shapes = (*shapes, (hf_config.linear_num_value_heads // tp_size, num_spec + 1,
                                hf_config.linear_value_head_dim + hf_config.linear_key_head_dim + 1))
        return shapes
"""),
])

# 5. new modules
shutil.copy("/opt/llm/runners/rssm/recoverssm_gdn.py", os.path.join(V, "model_executor/layers/mamba/gdn/recoverssm_gdn.py"))
shutil.copy("/opt/llm/runners/rssm/gdn_recoverssm.py", os.path.join(V, "v1/attention/backends/gdn_recoverssm.py"))
print("copied new modules")
