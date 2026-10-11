"""Capture Kolibri 1's Triton launches for the Zig CUDA port (tools/zig/triton_aot_manifest.py's Recorder).

Runs the Python engine's forward over the shapes the Zig forward will use: prompt chunks of several lengths, decode
rows for 1..16 streams, and verify chains of 2..4 rows per stream. Writes launches.json beside the per-case logits
(fp32, the reference the Zig forward must equal), then the caller builds aot.json + cubins with the manifest tool.

    TRITON_CACHE_DIR=<fresh> python capture_kolibri.py /opt/llm/models/kolibri-1 <out> --tf <TensorFold checkout>
    python <TensorFold>/tools/zig/triton_aot_manifest.py --cache <fresh> --launches <out>/launches.json --out <out>/aot
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--tf", required=True, help="a TensorFold checkout with tools/zig/triton_aot_manifest.py")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--wide", action="store_true", help="every router tile and tree width, divisible by 16 and not")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.tf) / "tools" / "zig"))
    from triton_aot_manifest import Recorder

    import torch

    from tensorfold.families.kolibri1.cuda.forward import Chain, Model
    from tensorfold.families.kolibri1.cuda.weights import load

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder().install()
    torch.cuda.set_device(0)
    with rec.scope("load"):
        w = load(a.model)
        m = Model(w, a.context, slots=16)
    g = torch.Generator().manual_seed(7)
    vocab = w.config.vocab

    def ids(n: int) -> list[int]:
        return torch.randint(100, vocab - 100, (n,), generator=g).tolist()

    calls = []                                          # every forward in order: the Zig test replays them

    def run(name: str | None, chains: list, prompt: bool, rows=None, detail: bool = True) -> None:
        with rec.scope(name or "fill", detail=detail and name is not None):
            logits = m.forward(chains, prompt=prompt, rows=rows)
        torch.cuda.synchronize()
        entry = {"name": name, "prompt": prompt, "rows": rows,
                 "chains": [{"slot": c.slot, "p0": c.p0, "tokens": list(c.tokens)} for c in chains]}
        if name is not None:
            f = out / f"{name}.logits.f32"
            f.write_bytes(logits.float().contiguous().cpu().numpy().tobytes())
            entry.update(file=f.name, shape=list(logits.shape))
        calls.append(entry)

    pos = {}
    for slot, n in enumerate((17, 300, 2000)):          # prompt chunks: small, one MoE tile, wide plans
        run(f"prompt{n}", [Chain(slot, 0, ids(n))], True)
        pos[slot] = n
    for slot in range(3, 16):                          # the other slots hold a short prompt each
        run(None, [Chain(slot, 0, ids(40))], True)
        pos[slot] = 40
    for streams in (1, 2, 3, 4, 16):                   # decode rows, one per stream
        chains = [Chain(s, pos[s], ids(1)) for s in range(streams)]
        run(f"decode{streams}", chains, False)
        for s in range(streams):
            pos[s] += 1
    for width in (2, 3, 4):                            # verify chains: `width` rows a stream, two streams
        chains = [Chain(s, pos[s], ids(width)) for s in (0, 1)]
        run(f"verify{width}", chains, False, rows=list(range(2 * width)))
        for s in (0, 1):
            pos[s] += width
    if a.wide:                                         # every row count class the Zig forward can hit
        for slot, n in zip(range(16), (16, 20, 32, 40, 48, 64, 72, 96, 100, 128, 160, 256, 300, 512, 600, 1024)):
            run(f"wprompt{n}", [Chain(slot, pos[slot], ids(n))], True)
            pos[slot] += n
        for streams, width in ((8, 1), (16, 1), (2, 8), (4, 4), (2, 16), (4, 8), (3, 8), (4, 12), (8, 8), (16, 2), (5, 7), (16, 4)):
            chains = [Chain(s, pos[s], ids(width)) for s in range(streams)]
            run(f"wverify{streams}x{width}", chains, False, rows=list(range(streams * width)))
            for s in range(streams):
                pos[s] += width
        for streams in (5, 8, 12, 16):
            chains = [Chain(s, pos[s], ids(1)) for s in range(streams)]
            run(f"wdecode{streams}", chains, False)
            for s in range(streams):
                pos[s] += 1
        for width in (1, 1, 2, 3):                      # one stream past 512 keys: the tree's shared items, W small
            long_slot = max(pos, key=pos.get)
            run(f"wlong{width}", [Chain(long_slot, pos[long_slot], ids(width))], False, rows=list(range(width)))
            pos[long_slot] += width
    rec.dump(out / "launches.json")
    (out / "calls.json").write_text(json.dumps({"context": a.context, "slots": 16, "calls": calls}, indent=1))
    print(json.dumps({"kernels": len(rec.kernels), "calls": len(calls)}))


if __name__ == "__main__":
    main()
