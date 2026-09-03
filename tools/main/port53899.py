import re, sys
F=sys.argv[1]; s=open(F).read()
# (a) forward -> forward_impl with the offload branch (PR hunk 8, adapted to main's fused GPU path)
old_fwd='''    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        # Keep num_reqs-dependent ID generation outside PIECEWISE CUDA graphs,
        # which dispatch only on the padded token count.
        # torch.compile requires the splitting op to write graph-owned storage.
        # Once compilation is removed, the op can return the IDs directly.
        ngram_ids = input_ids.new_empty(
            (input_ids.numel(), self.ngram_heads), dtype=torch.long
        )
        torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            ngram_ids,
            self.layer_name,
        )
        return self.ngram_embedding(ngram_ids).flatten(-2)
'''
new_fwd='''    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PORT53899 (jschmied 2026-09-03): vllm#53899's forward_impl on top of
        # main's fused-op forward. output_buffer is set only in the CPU offload
        # process, which is never captured by a CUDA graph.
        del hidden_states
        if output_buffer is None:
            # Keep num_reqs-dependent ID generation outside PIECEWISE CUDA graphs,
            # which dispatch only on the padded token count.
            ngram_ids = input_ids.new_empty(
                (input_ids.numel(), self.ngram_heads), dtype=torch.long
            )
            torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
                ngram_ids,
                self.layer_name,
            )
            return self.ngram_embedding(ngram_ids).flatten(-2)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        num_tokens = input_ids.reshape(-1).shape[0]
        quant_method = getattr(self.ngram_embedding, "quant_method", None)
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            output_dim = self.get_offload_output_dim(self.embedding_dim)
            output = output_buffer[:num_tokens, :output_dim]
            output.copy_(self.ngram_embedding(ngram_ids).flatten(-2))
            return output
        output = output_buffer[:num_tokens, : self.embedding_dim]
        torch.index_select(
            self.ngram_embedding.weight,
            0,
            ngram_ids.reshape(-1),
            out=output.reshape(-1, self.head_dim),
        )
        return output

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Keep quantized lookup results in their embedding storage dtype."""
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.dtype
        if isinstance(self._offload_quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            return torch.uint8
        if isinstance(self._offload_quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            return torch.float8_e4m3fn
        return default_dtype

    def get_offload_output_dim(self, default_dim: int) -> int:
        """Keep NVFP4 lookup rows packed while transferring them to the GPU."""
        quant_method = getattr(
            getattr(self, "ngram_embedding", None), "quant_method", None
        )
        if quant_method is None:
            quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            if default_dim % _NVFP4_BLOCK_SIZE:
                raise ValueError(
                    "NVFP4 PLE output dim must be a multiple of "
                    f"{_NVFP4_BLOCK_SIZE}, got {default_dim}"
                )
            return default_dim // 2 + default_dim // _NVFP4_BLOCK_SIZE
        return default_dim

    def initialize_dummy_offload_metadata(self, device: torch.device) -> None:
        """Initialize quantization metadata skipped by the dummy loader."""
        quant_method = self._offload_quant_method
        if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
        elif isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
            self.register_buffer(
                "_offload_weight_scale_2",
                torch.ones((), dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_offload_nvfp4_lut",
                torch.tensor(_FP4_VALUES, dtype=torch.float32, device=device),
                persistent=False,
            )
'''
assert s.count(old_fwd)==1, "fwd anchor"; s=s.replace(old_fwd,new_fwd)
# (c) retain-only branch at the top of NGram.load_weights
old_lw='''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,'''
new_lw='''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        # PORT53899: GPU workers retain only dequantization metadata. The CPU
        # process owns the embedding table and transfers quantized rows unchanged.
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            retained: set[str] = set()
            quant_method = self._offload_quant_method
            if isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
                for name, loaded_weight in weights:
                    if name != "ngram_embedding.weight_scale":
                        continue
                    self.register_buffer(
                        "_offload_weight_scale",
                        loaded_weight.to(
                            device=torch.accelerator.current_accelerator()
                        ),
                        persistent=False,
                    )
                    retained.add(name)
            elif isinstance(quant_method, Qwen4ExpPLENVFp4EmbeddingMethod):
                outer_scales: dict[int, torch.Tensor] = {}
                for name, loaded_weight in weights:
                    if not name.startswith(
                        "ngram_embedding.shard_"
                    ) or not name.endswith(".weight_scale_2"):
                        continue
                    shard_text = name[
                        len("ngram_embedding.shard_") : -len(".weight_scale_2")
                    ]
                    shard_index = int(shard_text)
                    outer_scales[shard_index] = loaded_weight
                    retained.add(name)
                if not retained:
                    raise ValueError(
                        "NVFP4 PLE offload checkpoint is missing its global scale"
                    )
                scale_2 = _get_shared_nvfp4_outer_scale(outer_scales).to(
                    device=torch.accelerator.current_accelerator()
                )
                self.register_buffer(
                    "_offload_weight_scale_2", scale_2, persistent=False
                )
                self.register_buffer(
                    "_offload_nvfp4_lut",
                    torch.tensor(
                        _FP4_VALUES, dtype=torch.float32, device=scale_2.device
                    ),
                    persistent=False,
                )
            else:
                for _ in weights:
                    pass
            return retained

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,'''
assert s.count(old_lw)==1, "lw anchor"; s=s.replace(old_lw,new_lw)
# (d) constructor block
old_init='''        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            prefix=f"{prefix}.ple_embedding",
            layer_name=prefix,
            quant_config=quant_config,
            params_dtype=model_config.dtype,
        )
'''
new_init='''        # PORT53899: the offload process builds the surrounding model on meta
        # while this subtree must own real CPU storage. GPU workers skip the
        # subclass constructor and retain only an empty IPC placeholder.
        with torch.device(PleOffloadLayer.get_target_device()):
            ple_embedding = Qwen4ExpNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                prefix=f"{prefix}.ple_embedding",
                layer_name=prefix,
                quant_config=quant_config,
                params_dtype=model_config.dtype,
            )
        if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
            ple_embedding._offload_quant_method = _get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ple_embedding.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            )
        self.ple_embedding: nn.Module = ple_embedding
'''
assert s.count(old_init)==1, "init anchor"; s=s.replace(old_init,new_init)
# (e) gate widening for ModelOpt mixed-precision (our fp8head checkpoint: F8 shards + weight_scale)
old_gate='''    if isinstance(quant_config, ModelOptNvFp4Config):
        if not quant_config.is_checkpoint_nvfp4_serialized:
            return None
'''
new_gate='''    # PLEGATE (jschmied): ModelOpt mixed-precision checkpoints (FP8 body + NVFP4
    # experts, e.g. lovedheart's FP8-mixed Flash-Next) ship the PLE as F8_E4M3
    # shards + one global `ngram_embedding.weight_scale`.
    if quant_config is not None and getattr(quant_config, "get_name", None) is not None:
        try:
            _qname = quant_config.get_name()
        except Exception:
            _qname = ""
        if _qname == "modelopt_mixed":
            return Qwen4ExpPLEFp8EmbeddingMethod()

    if isinstance(quant_config, ModelOptNvFp4Config):
        if not quant_config.is_checkpoint_nvfp4_serialized:
            return None
'''
assert s.count(old_gate)==1, "gate anchor"; s=s.replace(old_gate,new_gate)
open(F,"w").write(s); print("port applied")
