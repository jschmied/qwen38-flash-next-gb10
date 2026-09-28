#!/usr/bin/env python3
"""Extract named tensors from a checkpoint dir into one new safetensors file (pure python).
argv: <dir> <out.safetensors> <name> [<name> ...]. Uses model.safetensors.index.json to find shards."""
import json, struct, sys
d, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
hdr, blobs, off = {}, [], 0
for n in names:
    f = open(f"{d}/{wm[n]}", "rb"); k = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(k)); v = h[n]
    a, b = v["data_offsets"]; f.seek(8 + k + a); data = f.read(b - a)
    hdr[n] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + len(data)]}; blobs.append(data); off += len(data)
hdr["__metadata__"] = {"source_dir": d}
hj = json.dumps(hdr).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
with open(out, "wb") as o:
    o.write(struct.pack("<Q", len(hj))); o.write(hj)
    for b in blobs: o.write(b)
print("wrote", out, len(names), "tensors", off, "bytes")
