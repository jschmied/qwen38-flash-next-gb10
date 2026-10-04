"""A one-layer EAGLE-style drafter for Kolibri-1: the target's normed state plus the next token's embedding in,
the following state out, read by the target's own (frozen) head.

Trainable: fc (2D -> D), one attention block (RoPE, causal), one SwiGLU MLP. About 80M parameters at D = 2560.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DraftConfig:
    hidden: int = 2560
    heads: int = 20
    head_dim: int = 128
    ffn: int = 4096
    eps: float = 1e-6
    theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float) -> None:
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(n)), eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.w).to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox halves over the last dim of x [B, H, T, Dh] at positions pos [T]."""

    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    rot = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class Drafter(nn.Module):
    def __init__(self, cfg: DraftConfig = DraftConfig()) -> None:
        super().__init__()
        d, h, hd = cfg.hidden, cfg.heads, cfg.head_dim
        self.cfg = cfg
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.n1, self.n2, self.out_norm = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q, self.k, self.v = (nn.Linear(d, h * hd, bias=False) for _ in range(3))
        self.o = nn.Linear(h * hd, d, bias=False)
        self.gate, self.up = nn.Linear(d, cfg.ffn, bias=False), nn.Linear(d, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, d, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1 / math.sqrt(m.in_features))
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, feats: torch.Tensor, emb: torch.Tensor, pos: torch.Tensor,
                past: tuple[torch.Tensor, torch.Tensor] | None = None):
        """feats, emb [B, T, D] -> the next states [B, T, D] (normed like the target's), and the new (k, v)."""

        b, t, _ = feats.shape
        h, hd = self.cfg.heads, self.cfg.head_dim
        z = self.fc(torch.cat([emb, feats], -1))
        a = self.n1(z)
        q = rope(self.q(a).view(b, t, h, hd).transpose(1, 2), pos, self.cfg.theta)
        k = rope(self.k(a).view(b, t, h, hd).transpose(1, 2), pos, self.cfg.theta)
        v = self.v(a).view(b, t, h, hd).transpose(1, 2)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        causal = past is None
        if not causal and t > 1:
            raise ValueError("with a cache, one position a call")
        att = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        z = z + self.o(att.transpose(1, 2).reshape(b, t, h * hd))
        m = self.n2(z)
        z = z + self.down(F.silu(self.gate(m)) * self.up(m))
        return self.out_norm(z), (k, v)


    def rollout(self, feats: torch.Tensor, embs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Training-time test: step j (0-based) of every chain at once; chain t's step j reads its step j-1 output.

        feats [T, D] the target's states; embs[j] [T, D] the embedding of the token after step j's position.
        Chain t's step j sits at position t + j and attends to the target-fed rows <= t and to chain t's own steps."""

        t, d = feats.shape
        h, hd = self.cfg.heads, self.cfg.head_dim
        rows = torch.arange(t, device=feats.device)
        ks, vs, outs, f = [], [], [], feats
        for j, emb in enumerate(embs):
            z = self.fc(torch.cat([emb, f], -1))[None]
            a = self.n1(z)
            pos = rows + j
            q = rope(self.q(a).view(1, t, h, hd).transpose(1, 2), pos, self.cfg.theta)
            ks.append(rope(self.k(a).view(1, t, h, hd).transpose(1, 2), pos, self.cfg.theta))
            vs.append(self.v(a).view(1, t, h, hd).transpose(1, 2))
            first = rows[None, :] <= rows[:, None]                       # target-fed rows <= t
            own = torch.eye(t, dtype=torch.bool, device=feats.device)     # chain t's own earlier steps
            mask = torch.cat([first] + [own] * j, 1)
            att = F.scaled_dot_product_attention(q, torch.cat(ks, 2), torch.cat(vs, 2), attn_mask=mask)
            z = z + self.o(att.transpose(1, 2).reshape(1, t, h * hd))
            m = self.n2(z)
            z = z + self.down(F.silu(self.gate(m)) * self.up(m))
            f = self.out_norm(z)[0]
            outs.append(f)
        return outs


def params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
