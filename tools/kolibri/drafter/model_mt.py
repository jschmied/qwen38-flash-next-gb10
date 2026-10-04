"""A multi-layer-feature drafter for Kolibri-1 (EAGLE-3 style): ``taps`` target layers' states, fused to one, plus the
next token's embedding in; the following state out, read by the target's own (frozen) head. Later chain steps read
the drafter's own output in place of the fused target states. ``layers`` decoder layers (RoPE, causal) after one fc.

About 58M parameters a layer plus a 13M fc at D = 2560. A grown layer starts as an identity (zero output projections).
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
    layers: int = 1
    taps: int = 1


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


class Block(nn.Module):
    def __init__(self, cfg: DraftConfig) -> None:
        super().__init__()
        d, h, hd = cfg.hidden, cfg.heads, cfg.head_dim
        self.cfg = cfg
        self.n1, self.n2 = RMSNorm(d, cfg.eps), RMSNorm(d, cfg.eps)
        self.q, self.k, self.v = (nn.Linear(d, h * hd, bias=False) for _ in range(3))
        self.o = nn.Linear(h * hd, d, bias=False)
        self.gate, self.up = nn.Linear(d, cfg.ffn, bias=False), nn.Linear(d, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, d, bias=False)

    def qkv(self, z: torch.Tensor, pos: torch.Tensor):
        b, t, _ = z.shape
        h, hd = self.cfg.heads, self.cfg.head_dim
        a = self.n1(z)
        q = rope(self.q(a).view(b, t, h, hd).transpose(1, 2), pos, self.cfg.theta)
        k = rope(self.k(a).view(b, t, h, hd).transpose(1, 2), pos, self.cfg.theta)
        return q, k, self.v(a).view(b, t, h, hd).transpose(1, 2)

    def finish(self, z: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        b, t, _ = z.shape
        z = z + self.o(att.transpose(1, 2).reshape(b, t, -1))
        m = self.n2(z)
        return z + self.down(F.silu(self.gate(m)) * self.up(m))


class Drafter(nn.Module):
    def __init__(self, cfg: DraftConfig = DraftConfig()) -> None:
        super().__init__()
        d = cfg.hidden
        self.cfg = cfg
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.fuse = nn.Linear(cfg.taps * d, d, bias=False) if cfg.taps > 1 else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.out_norm = RMSNorm(d, cfg.eps)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1 / math.sqrt(m.in_features))
        for blk in self.blocks:
            nn.init.zeros_(blk.o.weight)
            nn.init.zeros_(blk.down.weight)

    def forward(self, feats: torch.Tensor, emb: torch.Tensor, pos: torch.Tensor, past: list | None = None):
        """feats, emb [B, T, D] -> the next states [B, T, D] (normed like the target's), and each layer's (k, v)."""

        if past is not None and feats.shape[1] > 1:
            raise ValueError("with a cache, one position a call")
        z = self.fc(torch.cat([emb, self.fused(feats)], -1))
        kvs = []
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv(z, pos)
            if past is not None:
                k, v = torch.cat([past[i][0], k], 2), torch.cat([past[i][1], v], 2)
            kvs.append((k, v))
            z = blk.finish(z, F.scaled_dot_product_attention(q, k, v, is_causal=past is None))
        return self.out_norm(z), kvs

    def fused(self, feats: torch.Tensor) -> torch.Tensor:
        """Target states [..., taps * D] fused to [..., D]; the drafter's own states [..., D] pass as they are."""

        return self.fuse(feats) if self.fuse is not None and feats.shape[-1] != self.cfg.hidden else feats

    def rollout(self, feats: torch.Tensor, embs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Training-time test: step j of every chain at once; chain t's step j reads its step j-1 output.

        feats [T, D] the target's states; embs[j] [T, D] the embedding of the token after step j's position.
        Chain t's step j sits at position t + j and attends to the target-fed rows <= t and to chain t's own steps."""

        t = feats.shape[0]
        feats = self.fused(feats)
        rows = torch.arange(t, device=feats.device)
        first = rows[None, :] <= rows[:, None]                       # target-fed rows <= t
        own = torch.eye(t, dtype=torch.bool, device=feats.device)     # chain t's own earlier steps
        ks = [[] for _ in self.blocks]
        vs = [[] for _ in self.blocks]
        outs, f = [], feats
        for j, emb in enumerate(embs):
            z = self.fc(torch.cat([emb, f], -1))[None]
            mask = torch.cat([first] + [own] * j, 1)
            for i, blk in enumerate(self.blocks):
                q, k, v = blk.qkv(z, rows + j)
                ks[i].append(k)
                vs[i].append(v)
                att = F.scaled_dot_product_attention(q, torch.cat(ks[i], 2), torch.cat(vs[i], 2), attn_mask=mask)
                z = blk.finish(z, att)
            f = self.out_norm(z)[0]
            outs.append(f)
        return outs


def load(ck: dict, layers: int | None = None) -> Drafter:
    """A drafter from a checkpoint (one-layer checkpoints' flat names included), grown to ``layers`` if asked."""

    cfg = DraftConfig(**{**{"layers": 1}, **ck["config"]})
    state = {}
    for name, t in ck["state"].items():
        flat = name.split(".")[0] in ("n1", "n2", "q", "k", "v", "o", "gate", "up", "down")
        state[("blocks.0." + name) if flat else name] = t
    grow = layers is not None and layers > cfg.layers
    if grow:
        cfg.layers = layers
    dr = Drafter(cfg)
    missing, unexpected = dr.load_state_dict(state, strict=not grow)
    if unexpected or (missing and not grow):
        raise ValueError(f"checkpoint does not fit: missing {missing[:4]}, unexpected {unexpected[:4]}")
    return dr


def params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
