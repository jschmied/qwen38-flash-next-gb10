"""model_block's visibility: block position j at anchor t may depend on context rows <= t and the token after t only."""

import torch

from model_block import BlockConfig, BlockDrafter, load


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


def test_spine_starts_as_identity_and_reads_only_its_predecessor():
    torch.manual_seed(1)
    cfg = BlockConfig(hidden=64, heads=4, head_dim=16, ffn=128, layers=3, taps=3, block=4)
    base = BlockDrafter(cfg).double()
    for p in base.parameters():
        torch.nn.init.normal_(p, std=0.2)
    ck = {"config": dict(cfg.__dict__), "state": base.state_dict()}
    dr = load(ck, spine_rank=8).double()
    T = 32
    feats = torch.randn(T, 3 * 64, dtype=torch.float64)
    first = torch.randn(3, 64, dtype=torch.float64)
    anchors = torch.tensor([5, 12, 20])
    out = dr(feats, first, anchors)
    assert torch.allclose(out, base(feats, first, anchors))  # zero-initialised: the loaded drafter's function, exactly
    for u in dr.spine_u:
        torch.nn.init.normal_(u.weight, std=0.2)
    out2 = dr(feats, first, anchors)
    assert not torch.allclose(out2[:, 1:], out[:, 1:])         # rows 1.. read their predecessors now
    f2 = feats.clone()
    f2[13:] += torch.randn_like(f2[13:])                       # still no leak from rows after an anchor
    out3 = dr(f2, first, anchors)
    assert torch.allclose(out3[1], out2[1]) and torch.allclose(out3[0], out2[0])
    solo = dr(feats, first[1:2], anchors[1:2])                 # nor across blocks
    assert torch.allclose(solo[0], out2[1])


if __name__ == "__main__":
    test_no_leak_and_isolation()
    test_spine_starts_as_identity_and_reads_only_its_predecessor()
    print("ok")


def test_chain_init_matches_chain_first_draft():
    """A block drafter built from a chain drafter computes, at position 0, exactly the chain's first draft."""

    from model_block import from_chain
    from model_mt import DraftConfig, Drafter

    torch.manual_seed(1)
    ccfg = DraftConfig(hidden=64, heads=4, head_dim=16, ffn=96, layers=1, taps=3)
    chain = Drafter(ccfg).double()
    for p in chain.parameters():
        torch.nn.init.normal_(p, std=0.2)
    dr = from_chain({"state": chain.state_dict()},
                    BlockConfig(hidden=64, heads=4, head_dim=16, ffn=128, layers=3, taps=3, block=4)).double()
    for layer in dr.layers[1:]:                                # identity at init: o/down zero
        assert not layer.o.weight.any() and not layer.down.weight.any()
    T = 40
    feats = torch.randn(T, 3 * 64, dtype=torch.float64)
    nxt = torch.randn(T, 64, dtype=torch.float64)           # embedding of the token after each row
    first = chain.rollout(feats, [nxt])[0]                    # chain step 1 at every row
    anchors = torch.tensor([3, 17, 30, 35])
    out = dr(feats, nxt[anchors], anchors, nxt=nxt)
    assert torch.allclose(out[:, 0], first[anchors], atol=1e-6)  # rope runs in fp32, float((out[:, 0] - first[anchors]).abs().max())
    # and the later rows still cannot see rows past the anchor
    f2 = feats.clone()
    f2[18:] += 1
    n2 = nxt.clone()
    n2[18:] += 1
    assert torch.allclose(dr(f2, nxt[anchors], anchors, nxt=n2)[1], out[1])


if __name__ == "__main__":
    test_chain_init_matches_chain_first_draft()
    print("chain ok")


def test_row0_chain_stays_exact_when_later_layers_train():
    """row0_chain: position 0 is the chain drafter whatever layers 1.. and the mask become."""

    from model_block import chain_params, from_chain
    from model_mt import DraftConfig, Drafter

    torch.manual_seed(2)
    chain = Drafter(DraftConfig(hidden=64, heads=4, head_dim=16, ffn=96, layers=1, taps=3)).double()
    for p in chain.parameters():
        torch.nn.init.normal_(p, std=0.2)
    cfg = BlockConfig(hidden=64, heads=4, head_dim=16, ffn=128, layers=3, taps=3, block=4, row0_chain=True)
    dr = from_chain({"state": chain.state_dict()}, cfg).double()
    frozen = {id(p) for p in chain_params(dr)}
    with torch.no_grad():                                       # "training" moves everything else
        for p in dr.parameters():
            if id(p) not in frozen:
                p.add_(torch.randn_like(p) * 0.3)
    T = 40
    feats = torch.randn(T, 3 * 64, dtype=torch.float64)
    nxt = torch.randn(T, 64, dtype=torch.float64)
    anchors = torch.tensor([3, 17, 30, 35])
    out = dr(feats, nxt[anchors], anchors, nxt=nxt)
    first = chain.rollout(feats, [nxt])[0]
    assert torch.allclose(out[:, 0], first[anchors], atol=1e-6), float((out[:, 0] - first[anchors]).abs().max())
    assert not torch.allclose(out[:, 1], dr.out_norm(torch.zeros(4, 64, dtype=torch.float64) + 1))  # rows 1.. are live


if __name__ == "__main__":
    test_row0_chain_stays_exact_when_later_layers_train()
    print("row0 ok")
