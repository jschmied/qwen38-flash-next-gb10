#!/usr/bin/env python3
"""Per-tensor diff of safetensors shards between two checkpoint dirs (pure python, read-only).
argv: <dirA> <dirB> <shard> [<shard> ...]. Prints ONE json: per shard the file sha256 of A, tensors only in A/B,
dtype/shape changes, and tensors whose bytes differ (name, dtype, shape, fraction of bytes differing)."""
import hashlib, json, struct, sys
A, B, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
def header(path):
    f = open(path, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, 8 + n, h
out = {}
for s in shards:
    fa, oa, ha = header(f"{A}/{s}"); fb, ob, hb = header(f"{B}/{s}")
    ta = {k: v for k, v in ha.items() if k != '__metadata__'}; tb = {k: v for k, v in hb.items() if k != '__metadata__'}
    r = {"only_A": sorted(set(ta) - set(tb)), "only_B": sorted(set(tb) - set(ta)), "meta_changed": [], "changed": [], "n": len(ta)}
    for k in sorted(set(ta) & set(tb)):
        va, vb = ta[k], tb[k]
        if va['dtype'] != vb['dtype'] or va['shape'] != vb['shape']:
            r["meta_changed"].append([k, va['dtype'], va['shape'], vb['dtype'], vb['shape']]); continue
        a0, a1 = va['data_offsets']; b0, b1 = vb['data_offsets']
        fa.seek(oa + a0); da = fa.read(a1 - a0); fb.seek(ob + b0); db = fb.read(b1 - b0)
        if da != db:
            diff = sum(1 for i in range(0, len(da), 4096) if da[i:i+4096] != db[i:i+4096]) / max(1, (len(da) + 4095) // 4096)
            r["changed"].append([k, va['dtype'], va['shape'], round(diff, 4)])
    h = hashlib.sha256(); fa.seek(0)
    for chunk in iter(lambda: fa.read(1 << 24), b''): h.update(chunk)
    r["sha256_A"] = h.hexdigest(); out[s] = r
print(json.dumps(out))
