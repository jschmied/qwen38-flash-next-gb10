"""A block drafter for Kolibri-1 (DFlash style): one pass drafts ``block`` tokens in parallel.

The block at anchor row t holds the token after t (known: Kolibri sampled it) at position t+1, then ``block - 1`` mask
embeddings at t+2.. . Every layer's attention sees the context rows <= t through keys and values projected from
Kolibri's fused tapped states (KV injection), and the block's own positions bidirectionally. Block position j's output
is read by Kolibri's own head as the token at t+2+j: ``block`` drafts a pass. Kolibri's embedding and head stay frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_mt import RMSNorm, rope


@dataclass
class BlockConfig:
    hidden: int = 2560
    heads: int = 20
    head_dim: int = 128
    ffn: int = 6144
    layers: int = 4
    taps: int = 3
    block: int = 4
    eps: float = 1e-6
    theta: float = 10000.0


class Layer(nn.Module):
    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        d, hd = cfg.hidden, cfg.heads * cfg.head_dim
        self.cfg = cfg
        self.n1, self.n2, self.nc = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q, self.k, self.v = (nn.Linear(d, hd, bias=False) for _ in range(3))
        self.ck, self.cv = nn.Linear(d, hd, bias=False), nn.Linear(d, hd, bias=False)   # context keys/values (injection)
        self.o = nn.Linear(hd, d, bias=False)
        self.gate, self.up = nn.Linear(d, cfg.ffn, bias=False), nn.Linear(d, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, d, bias=False)

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.cfg.heads, self.cfg.head_dim).transpose(0, 1)    # [H, T, Dh]

    def forward(self, z, pos, ctx, ctx_pos, mask):
        """z [Q, D] block rows at pos; ctx [T, D] fused context at ctx_pos; mask [Q, T + Q] (True: may attend)."""

        a, c = self.n1(z), self.nc(ctx)
        q = rope(self.heads(self.q(a))[None], pos, self.cfg.theta)
        k = torch.cat([rope(self.heads(self.ck(c))[None], ctx_pos, self.cfg.theta),
                       rope(self.heads(self.k(a))[None], pos, self.cfg.theta)], 2)
        v = torch.cat([self.heads(self.cv(c))[None], self.heads(self.v(a))[None]], 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])[0]
        z = z + self.o(att.transpose(0, 1).reshape(z.shape[0], -1))
        m = self.n2(z)
        return z + self.down(F.silu(self.gate(m)) * self.up(m))


class BlockDrafter(nn.Module):
    def __init__(self, cfg: BlockConfig = BlockConfig()) -> None:
        super().__init__()
        d = cfg.hidden
        self.cfg = cfg
        self.fuse = nn.Linear(cfg.taps * d, d, bias=False)
        self.mask = nn.Parameter(torch.zeros(d))
        self.inp = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.layers))
        self.out_norm = RMSNorm(d, cfg.eps)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1 / math.sqrt(m.in_features))
        for layer in self.layers:
            nn.init.zeros_(layer.o.weight)
            nn.init.zeros_(layer.down.weight)
        nn.init.normal_(self.mask, std=0.02)

    def forward(self, feats: torch.Tensor, first: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """feats [T, taps*D] Kolibri's tapped states of a window's rows; first [A, D] the embedding of the token after each
        anchor row; anchors [A] row indices (context rows <= anchor). Returns [A, block, D] states for the head."""

        cfg, dev = self.cfg, feats.device
        t, a, b = feats.shape[0], anchors.shape[0], cfg.block
        ctx = self.fuse(feats)
        rows = torch.arange(t, device=dev)
        inp = torch.cat([first[:, None], self.mask.to(first.dtype).expand(a, b - 1, -1)], 1)   # [A, B, D]
        z = self.inp(inp).reshape(a * b, -1)
        pos = (anchors[:, None] + 1 + torch.arange(b, device=dev)[None]).reshape(-1)
        owner = torch.arange(a, device=dev).repeat_interleave(b)                      # each query's block
        see_ctx = rows[None, :] <= anchors[owner][:, None]                              # [Q, T]
        see_blk = owner[:, None] == owner[None, :]                                      # [Q, Q]: own block, both ways
        mask = torch.cat([see_ctx, see_blk], 1)
        for layer in self.layers:
            z = layer(z, pos, ctx, rows, mask)
        return self.out_norm(z).view(a, b, -1)


def load(ck: dict) -> BlockDrafter:
    dr = BlockDrafter(BlockConfig(**ck["config"]))
    dr.load_state_dict(ck["state"])
    return dr
