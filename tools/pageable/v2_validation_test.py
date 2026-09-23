"""Review items 1+2 (2026-09-23): CheckpointTable must refuse incomplete/misshapen shard sets and never read
outside the mapping. Synthetic safetensors, real CUDA gather."""
import json, os, struct, tempfile, sys, numpy as np, torch
from vllm.v1.ple_offload.pageable import CheckpointTable, require_pageable_access

def write(d, shards, pad=7):   # shards: {index: nrows}; unaligned data start like the real checkpoint
    hdr, off, blobs = {}, 0, []
    for i, n in shards.items():
        a = (np.arange(n * 160, dtype=np.int64) * 7 + i * 131).astype(np.uint8)
        hdr[f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"] = {
            "dtype": "F8_E4M3", "shape": [n, 160], "data_offsets": [off, off + a.size]}
        off += a.size; blobs.append(a.tobytes())
    h = json.dumps(hdr).encode() + b" " * pad
    with open(os.path.join(d, "model-plefp8-00000.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); f.write(b"".join(blobs))

def expect_refusal(name, shards, num_rows, parts):
    with tempfile.TemporaryDirectory() as d:
        write(d, shards)
        try:
            CheckpointTable(d, num_rows, 160, parts)
        except ValueError as e:
            print(f"PASS refuse {name}: {str(e)[:90]}"); return True
        print(f"FAIL {name}: accepted"); return False

ok = True
require_pageable_access(0); print("PASS device guard accepts this GPU")
ok &= expect_refusal("missing shard", {0: 3, 1: 3, 3: 3}, 12, 4)
ok &= expect_refusal("8 rows for 12 requested", {0: 3, 1: 3, 2: 2}, 12, 4)
ok &= expect_refusal("short interior shard", {0: 3, 1: 2, 2: 3, 3: 3}, 12, 4)
ok &= expect_refusal("extra shard", {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}, 12, 4)
with tempfile.TemporaryDirectory() as d:
    write(d, {0: 3, 1: 3, 2: 3, 3: 2}, pad=5)      # 11 rows, last shard short = legal
    t = CheckpointTable(d, 11, 160, 4)
    ids = torch.tensor([0, 2, 3, 5, 9, 10, 11, 12, -1, 10**9], device="cuda")
    g = t.gather(ids).view(torch.uint8).cpu().numpy()
    ref = np.concatenate([(np.arange(n * 160, dtype=np.int64) * 7 + i * 131).astype(np.uint8).reshape(n, 160)
                          for i, n in {0: 3, 1: 3, 2: 3, 3: 2}.items()])
    good = np.array_equal(g[:6], ref[[0, 2, 3, 5, 9, 10]])
    zeros = not g[6:].any()
    print(("PASS" if good else "FAIL") + " in-range rows bit-exact (unaligned start, short last shard)")
    print(("PASS" if zeros else "FAIL") + " ids 11, 12, -1, 1e9 -> zero rows, no read outside the mapping")
    ok &= good and zeros
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
