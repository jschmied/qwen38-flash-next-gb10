#!/usr/bin/env python3
"""Micro-test for finding 225: GPU gather straight from an mmap'd file on GB10 (pageable memory access)."""
import mmap, os, sys, time, ctypes, json
import numpy as np, torch
PATH = sys.argv[1]; ROWS = os.path.getsize(PATH) // 160

sys.path.insert(0, "/opt/llm/runners")
from pageable_tensor import map_file_as_cuda
def wrap(path):
    t, fd, arr = map_file_as_cuda(path, 160)
    return fd, None, arr, t

res = {}
fd, mm, host, dev = wrap(PATH)
print("wrapped", dev.shape, dev.dtype, dev.device, hex(dev.data_ptr()), flush=True)
hostv = host.reshape(ROWS, 160)
g = torch.Generator().manual_seed(0)

def gather(ids_cpu):
    ids = ids_cpu.cuda()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = torch.index_select(dev, 0, ids)
    torch.cuda.synchronize(); return out, time.perf_counter() - t0

def drop():
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

# a. correctness, 480k random rows
ids = torch.randint(0, ROWS, (480_000,), generator=g)
out, _ = gather(ids)
ref = torch.from_numpy(hostv[ids.numpy()])
res["a_bitexact_480k"] = bool(torch.equal(out.cpu(), ref))
# b/c: 16 rows warm vs cold, 20 trials each
for label, cold in (("c_cold16", True), ("b_warm16", False)):
    ts = []
    for k in range(20):
        ids = torch.randint(0, ROWS, (16,), generator=g)
        if cold: drop()
        else: gather(ids)
        ts.append(gather(ids)[1] * 1e3)
    res[label + "_ms"] = [round(min(ts), 3), round(float(np.median(ts)), 3), round(max(ts), 3)]
# d: cold 480k (a 30k-token prefill), then warm repeat of the same ids, then a CPU WILLNEED-prefetch arm
ids = torch.randint(0, ROWS, (480_000,), generator=g)
drop(); _, t = gather(ids); res["d_cold480k_s"] = round(t, 3)
_, t = gather(ids); res["d_warm480k_s"] = round(t, 3)
drop()
t0 = time.perf_counter()
pages = np.unique((ids.numpy().astype(np.int64) * 160) // 4096)
for p in pages: os.posix_fadvise(fd, int(p) * 4096, 8192, os.POSIX_FADV_WILLNEED)
tp = time.perf_counter() - t0
_, t = gather(ids); res["d_prefetch_issue_s"] = round(tp, 3); res["d_after_willneed_s"] = round(t, 3)
res["torch_cuda_mem_alloc_MiB"] = torch.cuda.memory_allocated() // 2**20
print(json.dumps(res))
