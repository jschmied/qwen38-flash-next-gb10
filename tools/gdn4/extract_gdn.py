"""Extract the GDN projection tensors (in_proj_qkv, in_proj_z, out_proj; BF16) of RadixArk Qwen3.8-Flash-Next-NVFP4
into one safetensors file, byte-exact, plus a provenance json. No libraries beyond the stdlib."""
import json, struct, os, re, hashlib
SRC = '/mnt/bulk/gb10/models/qwen38-flash-next-nvfp4'
OUT = '/mnt/bulk/gb10/tmp/gdn-proj-bf16'
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(SRC + '/model.safetensors.index.json'))['weight_map']
pat = re.compile(r'layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$')
names = sorted((k for k in idx if pat.search(k)), key=lambda k: (int(pat.search(k).group(1)), k))
hdrs = {}
def header(f):
    if f not in hdrs:
        b = open(SRC + '/' + f, 'rb'); n = struct.unpack('<Q', b.read(8))[0]; hdrs[f] = (json.loads(b.read(n)), 8 + n)
    return hdrs[f]
meta, off, plan = {}, 0, []
for k in names:
    h, base = header(idx[k]); t = h[k]; a, e = t['data_offsets']
    meta[k] = {'dtype': t['dtype'], 'shape': t['shape'], 'data_offsets': [off, off + e - a]}
    plan.append((k, idx[k], base + a, e - a)); off += e - a
hj = json.dumps(meta, separators=(',', ':')).encode(); hj += b' ' * ((8 - len(hj) % 8) % 8)
prov = {'source_dir': SRC, 'source_repo': 'RadixArk Qwen3.8-Flash-Next-NVFP4 (linear_attn in ignore -> original BF16)',
        'tensors': []}
with open(OUT + '/gdn_proj_bf16.safetensors', 'wb') as o:
    o.write(struct.pack('<Q', len(hj))); o.write(hj)
    for k, f, start, size in plan:
        with open(SRC + '/' + f, 'rb') as s:
            s.seek(start); buf = s.read(size)
        assert len(buf) == size
        o.write(buf)
        prov['tensors'].append({'name': k, 'shard': f, 'offset': start, 'bytes': size, 'sha256': hashlib.sha256(buf).hexdigest()})
json.dump(prov, open(OUT + '/SOURCE.json', 'w'), indent=1)
print(len(names), 'tensors', off / 2**30, 'GiB')
