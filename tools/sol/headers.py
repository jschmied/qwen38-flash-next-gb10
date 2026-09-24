import json, struct, re, collections, sys
D = "/opt/llm/models/qwen38-flash-next-mtpfp4"
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
BYTES = {"F8_E4M3": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F8_E5M2": 1, "BOOL": 1}
rows = []
for f in sorted(set(wm.values())):
    with open(f"{D}/{f}", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        nb = BYTES[v["dtype"]]
        for s in v["shape"]: nb *= s
        rows.append((k, v["dtype"], v["shape"], nb))
json.dump(rows, open("tensors.json", "w"))
pat = collections.defaultdict(lambda: [0, 0, set()])
for k, dt, sh, nb in rows:
    p = re.sub(r"\.\d+\.", ".#.", k); p = re.sub(r"\.\d+\.", ".#.", p)
    pat[p][0] += nb; pat[p][1] += 1; pat[p][2].add(dt)
tot = sum(r[3] for r in rows)
print(f"tensors {len(rows)}, total {tot/2**30:.2f} GiB")
for p, (nb, cnt, dts) in sorted(pat.items(), key=lambda kv: -kv[1][0])[:60]:
    print(f"{nb/2**30:9.3f} GiB  n={cnt:6d}  {','.join(sorted(dts)):12s} {p}")
