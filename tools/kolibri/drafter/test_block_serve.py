"""The serving block drafter (tensorfold kolibri1 cuda/block_drafter.py) against model_block + eval_block's predecessor
loop, in float64 on CPU: same states, same drafts, with context added in chunks as the decoder adds kept rows."""

import json

import torch

from model_block import BlockConfig, BlockDrafter, load, predecessor


def test_serving_pass_equals_the_training_model(tmp_path):
    from tensorfold.families.kolibri1.cuda.block_drafter import BlockDrafter as Serve

    torch.manual_seed(3)
    d, V, T = 64, 50, 37
    cfg = BlockConfig(hidden=d, heads=4, head_dim=16, ffn=96, layers=3, taps=3, block=4, chain_ctx=True,
                      row0_chain=True, pred_rank=8)
    base = BlockDrafter(cfg)
    ck0 = {"config": dict(cfg.__dict__), "state": base.state_dict()}
    dr = load(ck0, pred_rank=8, spine_rank=8).double()
    for p in dr.parameters():                                     # zero-init parts too: every path carries signal
        torch.nn.init.normal_(p, std=0.15)
    ck = {"config": dict(dr.cfg.__dict__), "state": dr.state_dict(), "taps": [44, 47, 49]}
    torch.save(ck, tmp_path / "d.pt")
    vocab = list(range(0, V, 2)) + [1, 3]
    (tmp_path / "v.json").write_text(json.dumps(vocab))
    embed = torch.randn(V, d, dtype=torch.float64)
    head = torch.randn(V, d, dtype=torch.float64)
    feats = torch.randn(T, 3 * d, dtype=torch.float64)
    tok = torch.randint(0, V, (T + 1,))

    with torch.no_grad():                                         # reference: anchor T-1, rows 0..T-1 as context
        st = dr(feats, embed[tok[1:][-1:]], torch.tensor([T - 1]), nxt=embed[tok[1:]])
        vt = torch.tensor(vocab)
        cols = [(st[:, 0] @ head[vt].T).argmax(-1)]
        states = [st[:, 0]]
        for j in range(1, cfg.block):
            sj = predecessor(dr, st[:, j], embed[vt[cols[-1]]])
            states.append(sj)
            cols.append((sj @ head[vt].T).argmax(-1))
        want = vt[torch.stack(cols, 1)][0].tolist()

    sv = Serve(tmp_path / "d.pt", embed, head, vocab=tmp_path / "v.json", dtype=torch.float64, depth=4)
    taps = [feats[:, i * d:(i + 1) * d] for i in range(3)]
    sv.add(0, list(range(0, 20)), tok[1:][0:20].tolist(), [t[0:20] for t in taps])
    sv.chain(0)                                                   # a pass in between: its scratch rows must not leak
    sv.add(0, list(range(20, T)), tok[1:][20:T].tolist(), [t[20:T] for t in taps])
    got = sv.chain(0)
    assert got == want, (got, want)
    for a, b in zip(sv.states, states):
        assert torch.allclose(a[0], b[0], atol=1e-9), float((a[0] - b[0]).abs().max())
    assert sv.chain(0) == want                                    # and a repeated pass gives the same drafts


def test_one_pass_for_many_streams_equals_each_alone(tmp_path):
    from tensorfold.families.kolibri1.cuda.block_drafter import BlockDrafter as Serve

    torch.manual_seed(5)
    d, V = 64, 50
    cfg = BlockConfig(hidden=d, heads=4, head_dim=16, ffn=96, layers=3, taps=3, block=4, chain_ctx=True,
                      row0_chain=True, pred_rank=8)
    dr = load({"config": dict(cfg.__dict__), "state": BlockDrafter(cfg).state_dict()}, pred_rank=8, spine_rank=8)
    for p in dr.parameters():
        torch.nn.init.normal_(p, std=0.15)
    torch.save({"config": dict(dr.cfg.__dict__), "state": dr.state_dict(), "taps": [44, 47, 49]}, tmp_path / "d.pt")
    embed, head = torch.randn(V, d, dtype=torch.float64), torch.randn(V, d, dtype=torch.float64)
    sv = Serve(tmp_path / "d.pt", embed, head, dtype=torch.float64, depth=4)
    for sid, T in ((0, 23), (1, 41), (2, 9)):                    # different context lengths
        feats = torch.randn(T, 3 * d, dtype=torch.float64)
        sv.add(sid, list(range(T)), torch.randint(0, V, (T,)).tolist(), [feats[:, i * d:(i + 1) * d] for i in range(3)])
    alone = {sid: sv.chain(sid) for sid in (0, 1, 2)}
    assert sv.chain_many([0, 1, 2]) == alone
    assert sv.chain_many([2, 7, 0]) == {2: alone[2], 7: [], 0: alone[0]}   # an unknown stream drafts nothing


def test_release_directory_drafts_like_the_checkpoint(tmp_path):
    import subprocess
    import sys

    from tensorfold.families.kolibri1.cuda.block_drafter import BlockDrafter as Serve, is_block

    torch.manual_seed(7)
    d, V, T = 64, 50, 30
    cfg = BlockConfig(hidden=d, heads=4, head_dim=16, ffn=96, layers=3, taps=3, block=4, chain_ctx=True,
                      row0_chain=True, pred_rank=8)
    dr = load({"config": dict(cfg.__dict__), "state": BlockDrafter(cfg).state_dict()}, pred_rank=8, spine_rank=8)
    with torch.no_grad():                                         # bf16-exact weights: the release stores bf16
        for p in dr.parameters():
            p.copy_(torch.randn_like(p).mul(0.15).to(torch.bfloat16).float())
    torch.save({"config": dict(dr.cfg.__dict__), "state": dr.state_dict(), "taps": [44, 47, 49], "step": 9,
                "tokens": 99}, tmp_path / "d.pt")
    (tmp_path / "v.json").write_text(json.dumps(list(range(0, V, 2))))
    subprocess.run([sys.executable, "export_block.py", tmp_path / "d.pt", tmp_path / "v.json", tmp_path / "rel"],
                   check=True, capture_output=True)
    assert is_block(tmp_path / "rel") and is_block(tmp_path / "d.pt")
    meta = json.loads((tmp_path / "rel" / "config.json").read_text())
    assert meta["serving"] == {"depth": 2, "quant": "fp8", "head": "nvfp4"} and meta["training_steps"] == 9
    embed, head = torch.randn(V, d, dtype=torch.float64), torch.randn(V, d, dtype=torch.float64)
    feats = torch.randn(T, 3 * d, dtype=torch.float64)
    toks = torch.randint(0, V, (T,)).tolist()
    out = []
    for src, kw in ((tmp_path / "d.pt", {"vocab": tmp_path / "v.json"}), (tmp_path / "rel", {})):
        sv = Serve(src, embed, head, dtype=torch.float64, depth=4, **kw)
        sv.add(0, list(range(T)), toks, [feats[:, i * d:(i + 1) * d] for i in range(3)])
        out.append(sv.chain(0))
    assert out[0] == out[1] and len(out[0]) == 4
    assert Serve(tmp_path / "rel", embed, head, dtype=torch.float64).depth == 2    # the release's serving default
