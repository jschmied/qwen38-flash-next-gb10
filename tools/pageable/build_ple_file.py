#!/usr/bin/env python3
"""Rewrite the 128 plefp8 shard tensors into one contiguous, page-aligned file in logical row order."""
import json, os, struct, sys, glob
src, dst = sys.argv[1], sys.argv[2]
shards = {}
for f in sorted(glob.glob(os.path.join(src, "model-plefp8-*.safetensors"))):
    with open(f, "rb") as b:
        n = struct.unpack("<Q", b.read(8))[0]; h = json.loads(b.read(n))
    for k, v in h.items():
        if ".ngram_embedding.shard_" in k and k.endswith(".weight"):
            i = int(k.split(".shard_")[1].split(".")[0])
            s, e = v["data_offsets"]; shards[i] = (f, 8 + n + s, e - s, v["shape"])
assert sorted(shards) == list(range(128)), sorted(shards)[:5]
tmp = dst + ".partial"
out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
pos = 0
for i in range(128):
    f, off, ln, shape = shards[i]
    assert shape == [2500012, 160] and ln == 2500012 * 160
    fd = os.open(f, os.O_RDONLY)
    done = 0
    while done < ln:
        done += os.copy_file_range(fd, out, ln - done, off + done, pos + done)
    os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED); os.close(fd)
    os.fdatasync(out); os.posix_fadvise(out, pos, ln, os.POSIX_FADV_DONTNEED)
    pos += ln
    if i % 16 == 15: print(f"shard {i+1}/128 {pos/2**30:.1f} GiB", flush=True)
os.close(out); os.rename(tmp, dst)
print("DONE", dst, pos, "bytes")
