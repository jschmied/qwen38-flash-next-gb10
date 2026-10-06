"""model_block's visibility: block position j at anchor t may depend on context rows <= t and the token after t only."""

import torch

from model_block import BlockConfig, BlockDrafter


def test_no_leak_and_isolation():
    torch.manual_seed(0)
    cfg = BlockConfig(hidden=64, heads=4, head_dim=16, ffn=128, layers=2, taps=3, block=4)
    dr = BlockDrafter(cfg).double()
    for p in dr.parameters():                                  # o/down start at zero: give every path a signal
        torch.nn.init.normal_(p, std=0.2)
    T = 32
    feats = torch.randn(T, 3 * 64, dtype=torch.float64)
    first = torch.randn(3, 64, dtype=torch.float64)
    anchors = torch.tensor([5, 12, 20])
    out = dr(feats, first, anchors)
    # rows after an anchor (including t+1, whose state does not exist when drafting) must not move its block
    f2 = feats.clone()
    f2[13:] += torch.randn_like(f2[13:])
    out2 = dr(f2, first, anchors)
    assert torch.allclose(out[1], out2[1]) and torch.allclose(out[0], out2[0])
    assert not torch.allclose(out[2], out2[2])                 # anchor 20 does see row 13..20
    # a row at the anchor itself is visible
    f3 = feats.clone()
    f3[12] += 1
    assert not torch.allclose(dr(f3, first, anchors)[1], out[1])
    # another block's first token must not leak into this block
    g = first.clone()
    g[0] += 1
    out4 = dr(feats, g, anchors)
    assert torch.allclose(out4[1], out[1]) and torch.allclose(out4[2], out[2])
    assert not torch.allclose(out4[0], out[0])
    # a block alone equals the block among others
    solo = dr(feats, first[1:2], anchors[1:2])
    assert torch.allclose(solo[0], out[1])


if __name__ == "__main__":
    test_no_leak_and_isolation()
    print("ok")
