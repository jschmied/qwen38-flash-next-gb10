# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM protocol for the Qwen4Exp PLE short-conv layer (FNRSSM phase 2, jschmied 2026-09-25, local).

When the GDN layers run RecoverSSM, the runner resets the shared num_accepted_tokens to 1 after each commit (align
mode). The PLE dilated short conv keeps an extended window in one state block and reads it at num_accepted - 1, so
it must follow the same protocol: every active decode row takes the spec path, the conv reads at offset 0, and after
sampling the accepted window is compacted to the front (generic Kimi-K3 compaction kernel; in align mode it also
writes the block-boundary window).
"""
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata, RecoverSSMPostprocessMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)


class _PleConvCommit:
    def __init__(self, conv_states, spec_query_len: int, max_num_reqs: int):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        if not is_conv_state_dim_first():
            conv_states = [s.transpose(-1, -2) for s in conv_states]   # -> [blocks, C, W]
        self.conv_states = list(conv_states)
        ref = self.conv_states[0]
        dev = ref.device
        self.conv_dim, conv_len = ref.shape[1], ref.shape[2]
        self.hist = conv_len - spec_query_len + 1
        assert self.hist > 0
        self.spec_query_len = spec_query_len
        t = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
        self.base = t([s.data_ptr() for s in self.conv_states])
        self.bstride = t([s.stride(0) for s in self.conv_states])
        self.dstride = t([s.stride(1) for s in self.conv_states])
        self.tstride = t([s.stride(2) for s in self.conv_states])
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len = z(), z(), z(), z()

    def commit(self, num_accepted_tokens, state_indices, query_start_loc, request_indices=None,
               block_table=None, num_computed_tokens=None, mamba_block_size=None):
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel, _prepare_commit_plan_kernel
        batch = state_indices.shape[0]
        if batch == 0:
            return
        bt = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens, request_indices, state_indices, query_start_loc, block_table, num_computed_tokens,
            self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len, NULL_BLOCK_ID, mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1, num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0, state_indices.stride(0),
            query_start_loc.stride(0), bt[0], bt[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len, num_warps=1)
        _compact_conv_state_kernel[(triton.cdiv(self.conv_dim, 256), batch, len(self.conv_states))](
            self.conv_states[0], self.base, self.bstride, self.dstride, self.tstride, state_indices,
            self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len, NULL_BLOCK_ID, self.conv_dim, self.hist,
            state_indices.stride(0), BLOCK_D=256, BLOCK_HISTORY=triton.next_power_of_2(self.hist),
            ALIGN_MODE=block_table is not None, num_warps=4)


@dataclass
class PleRecoverSSMCommitMetadata:
    state_indices: torch.Tensor     # [n]
    query_start_loc: torch.Tensor   # [n + 1]
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


@dataclass
class PleRecoverSSMMetadata(PleShortConvAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: PleRecoverSSMCommitMetadata | None = None
    recoverssm_context: Any = field(default=None, repr=False, compare=False)

    def commit_recoverssm_state(self, num_accepted_tokens):
        c = self.recoverssm_commit
        if c is None:
            return None
        n = self.num_spec_decodes
        self.recoverssm_context.commit(num_accepted_tokens, c.state_indices[:n], c.query_start_loc[: n + 1],
                                       request_indices=c.request_indices, block_table=c.block_table,
                                       num_computed_tokens=c.num_computed_tokens, mamba_block_size=c.block_size)
        if c.block_table is None:
            return None
        return RecoverSSMPostprocessMetadata(num_spec_decodes=n, request_indices=c.request_indices,
                                             block_table=c.block_table, num_computed_tokens=c.num_computed_tokens,
                                             block_size=c.block_size)


class PleRecoverSSMMetadataBuilder(PleShortConvAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._ones = torch.ones(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device)
        self._ctx = None
        self._logged = False

    def _get_ctx(self):
        if self._ctx is None:
            fc = self.vllm_config.compilation_config.static_forward_context
            convs = [fc[name].kv_cache[0] for name in self.layer_names]
            self._ctx = _PleConvCommit(convs, 1 + self.num_spec, self.vllm_config.scheduler_config.max_num_seqs)
            logger.warning("FNRSSM PLE commit context: %d layers, history %d", len(convs), self._ctx.hist)
        return self._ctx

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, *, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, **kwargs):  # type: ignore[override]
        m = common_attn_metadata
        spec_mask_cpu = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            assert m.is_prefilling is not None and m.is_prefilling.device.type == "cpu"
            active_decode = (~m.is_prefilling[: m.num_reqs]) & (m.query_start_loc_cpu.diff() > 0)
            spec_mask_cpu = (num_decode_draft_tokens_cpu[: m.num_reqs] >= 0) | active_decode
            drafts = torch.full_like(num_decode_draft_tokens_cpu, -1)
            drafts[: m.num_reqs] = torch.where(spec_mask_cpu, 1, -1).to(drafts.dtype)
            num_decode_draft_tokens_cpu = drafts
        meta = super().build(common_prefix_len, m, fast_build, num_accepted_tokens=num_accepted_tokens,
                             num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu, **kwargs)
        if not isinstance(meta, PleShortConvAttentionMetadata) or meta.num_spec_decodes == 0:
            return meta
        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        n = meta.num_spec_decodes
        base["num_accepted_tokens"] = self._ones[:n]
        rows = spec_mask_cpu.nonzero().flatten()
        contiguous = bool(rows.numel() == n and int(rows[-1]) == n - 1)
        req_idx = None if contiguous else rows.to(torch.int32).to(m.query_start_loc.device, non_blocking=True)
        align = self.vllm_config.cache_config.mamba_cache_mode == "align"
        commit = PleRecoverSSMCommitMetadata(
            state_indices=meta.spec_state_indices_tensor, query_start_loc=meta.spec_query_start_loc,
            request_indices=req_idx, block_table=m.block_table_tensor if align else None,
            num_computed_tokens=m.compute_num_computed_tokens() if align else None,
            block_size=self.kv_cache_spec.block_size if align else None)
        if not self._logged:
            self._logged = True
            logger.warning("FNRSSM PLE RecoverSSM path taken: %d spec rows, align=%s", n, align)
        return PleRecoverSSMMetadata(**base, recoverssm_commit=commit, recoverssm_context=self._get_ctx())


class PleRecoverSSMAttentionBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PLE_SHORT_CONV_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[PleRecoverSSMMetadataBuilder]:
        return PleRecoverSSMMetadataBuilder
