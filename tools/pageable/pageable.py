"""PROTOTYPE (finding 225): PLE n-gram table read by the GPU straight from an mmap'd file.

For GPUs with pageable memory access (GB10: CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES=1)
a kernel can dereference an ordinary host virtual address. The table then needs no pin, no swap, no worker
process and no staging copy. GPU faults on non-resident pages are serviced one page at a time (~0.16 ms each),
so a CPU thread prefetches the step's rows from the same input ids the model sees. The prefetch is a pure
performance hint: the GPU gather is correct whether or not it has finished.

Enabled by VLLM_PLE_PAGEABLE_FILE=<contiguous fp8 table, rows in logical order>, with VLLM_PLE_CPU_OFFLOAD=0.
"""
import ctypes
import mmap
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MADV_RANDOM = 1
_PAGE = 4096


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


_CU_PAGEABLE_MEMORY_ACCESS = 88
_CU_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100


def require_pageable_access(device_id: int) -> None:
    """Refuse unless the GPU can dereference ordinary pageable host memory through the host page tables.
    Being integrated is not sufficient, and a missing attribute is a refusal, not an acceptance."""
    cu = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    if cu.cuInit(0) != 0 or cu.cuDeviceGet(ctypes.byref(dev), device_id) != 0:
        raise RuntimeError("PLE pageable: cannot query the CUDA driver")
    for name, attr in (("PAGEABLE_MEMORY_ACCESS", _CU_PAGEABLE_MEMORY_ACCESS),
                       ("PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
                        _CU_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES)):
        v = ctypes.c_int(0)
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        if rc != 0 or v.value != 1:
            raise RuntimeError(f"PLE pageable: device {device_id} lacks CU_DEVICE_ATTRIBUTE_{name} "
                               f"(rc={rc}, value={v.value}); the GPU cannot read the mapped table")


# One mapping per process; the prefetcher reads it back from here.
MAPPING: dict = {}
_KEEP: list = []


def _cuda_view_of_host(addr: int, rows: int, cols: int, device_id: int) -> torch.Tensor:
    # torch.as_tensor(__cuda_array_interface__) refuses unregistered host pointers;
    # DLPack with an explicit kDLCUDA device does not check the pointer.
    shape = (ctypes.c_int64 * 2)(rows, cols)
    mt = _DLManagedTensor()
    mt.dl_tensor = _DLTensor(addr, _DLDevice(2, device_id), 2, _DLDataType(1, 8, 1), shape, None, 0)
    mt.manager_ctx = None
    mt.deleter = None
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    cap = new(ctypes.addressof(mt), b"dltensor", None)
    _KEEP.extend([shape, mt])
    return torch.utils.dlpack.from_dlpack(cap)


def map_table(path: str, num_rows: int, row_bytes: int) -> torch.Tensor:
    """Map the table read-only and return a float8_e4m3fn "cuda" view of its first num_rows rows."""
    require_pageable_access(torch.cuda.current_device())
    size = os.path.getsize(path)
    if size < num_rows * row_bytes:
        raise ValueError(f"{path}: {size} bytes < {num_rows} rows x {row_bytes}")
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
    host = np.frombuffer(mm, dtype=np.uint8)
    addr = host.__array_interface__["data"][0]
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), _MADV_RANDOM) != 0:
        logger.warning("PLE pageable: madvise(MADV_RANDOM) failed, errno %d", ctypes.get_errno())
    device_id = torch.cuda.current_device()
    view = _cuda_view_of_host(addr, size // row_bytes, row_bytes, device_id)[:num_rows]
    _KEEP.extend([mm, host])
    MAPPING.update(fd=fd, host=host.reshape(-1, row_bytes)[:num_rows], row_bytes=row_bytes)
    logger.info("PLE pageable: mapped %s (%.2f GiB, %d rows) at 0x%x, GPU reads it in place, nothing pinned",
                path, size / 2**30, num_rows, addr)
    return view.view(torch.float8_e4m3fn)


class PagePrefetcher:
    """Touch the pages holding this step's PLE rows from CPU threads while the GPU runs."""

    SLOTS = 4

    def __init__(self, model, device, input_ids_source, query_start_loc_source, ngram_context_source,
                 max_tokens: int, max_seqs: int) -> None:
        self.device = device
        self.table = TABLE.get("t")
        if self.table is None:
            self.host = MAPPING["host"]
            self.fd = MAPPING["fd"]
            self.row_bytes = MAPPING["row_bytes"]
            self.num_rows = self.host.shape[0]
        else:
            self.num_rows = self.table.num_rows
        self.sources = (input_ids_source, query_start_loc_source, ngram_context_source)
        self.layers = []
        for m in model.modules():
            if type(m).__name__ == "Qwen4ExpNGramEmbedding":
                ns = SimpleNamespace(
                    layer_multipliers=m.layer_multipliers.cpu(),
                    ngram_heads_vocab_sizes=m.ngram_heads_vocab_sizes.cpu(),
                    ngram_heads_offsets=m.ngram_heads_offsets.cpu(),
                    eos_token_id=m.eos_token_id, ngram_size=m.ngram_size,
                    heads_per_ngram=m.heads_per_ngram,
                    _shift_precompute=type(m)._shift_precompute,
                    _shift_apply=type(m)._shift_apply,
                )
                self.layers.append((m, ns))
        if not self.layers:
            raise RuntimeError("PLE pageable: no Qwen4ExpNGramEmbedding in the model")
        ctx_len = ngram_context_source.shape[1]
        self.slots = [dict(
            ids=torch.empty(max_tokens, dtype=input_ids_source.dtype).pin_memory(),
            qsl=torch.empty(max_seqs + 1, dtype=query_start_loc_source.dtype).pin_memory(),
            ctx=torch.empty(max_seqs, ctx_len, dtype=ngram_context_source.dtype).pin_memory(),
        ) for _ in range(self.SLOTS)]
        self.free = queue.Queue()
        for i in range(self.SLOTS):
            self.free.put(i)
        self.work: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(64, thread_name_prefix="ple-touch")
        self.checks_left = 3
        self.row_checks_left = 6   # gathered rows == file bytes, on real MIXED prefill+decode steps
        self.stats = dict(steps=0, skipped=0, rows=0, ms=0.0, lag_ms=0.0, big=0)
        self.thread = threading.Thread(target=self._loop, name="ple-prefetch", daemon=True)
        self.thread.start()
        logger.info("PLE pageable: prefetcher up, %d PLE layer(s), %d slots, 64 touch threads",
                    len(self.layers), self.SLOTS)

    def prepare_forward(self, num_reqs: int, num_tokens: int, dummy_run: bool) -> None:
        if dummy_run:
            return
        ids_src, qsl_src, ctx_src = self.sources
        if self.checks_left > 0:
            self._check_ids(num_reqs, num_tokens)
        if self.row_checks_left > 0 and self.table is not None and num_reqs >= 2:
            self._check_rows(num_reqs, num_tokens)
        try:
            slot = self.free.get_nowait()
        except queue.Empty:
            self.stats["skipped"] += 1   # prefetch is a hint; the gather stays correct
            return
        s = self.slots[slot]
        s["ids"][:num_tokens].copy_(ids_src[:num_tokens], non_blocking=True)
        s["qsl"][: num_reqs + 1].copy_(qsl_src[: num_reqs + 1], non_blocking=True)
        s["ctx"][:num_reqs].copy_(ctx_src[:num_reqs], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        self.work.put((slot, ev, num_reqs, num_tokens, time.perf_counter()))

    def _cpu_rows(self, ids, qsl, ctx):
        out = []
        for m, ns in self.layers:
            r = type(m).compute_ngram_ids(ns, ids.long(), qsl.long(), ctx.long())
            out.append(r.reshape(-1).numpy())
        return np.concatenate(out)

    def _check_ids(self, num_reqs: int, num_tokens: int) -> None:
        # Void check: the CPU ids must equal the ids the GPU gathers, or the prefetch warms the wrong pages.
        self.checks_left -= 1
        ids_src, qsl_src, ctx_src = self.sources
        m = self.layers[0][0]
        gpu = m.compute_ngram_ids(ids_src[:num_tokens], qsl_src[: num_reqs + 1], ctx_src[:num_reqs]).cpu()
        cpu = self._cpu_rows(ids_src[:num_tokens].cpu(), qsl_src[: num_reqs + 1].cpu(), ctx_src[:num_reqs].cpu())
        ok = np.array_equal(gpu.reshape(-1).numpy()[: cpu.size // len(self.layers)], cpu[: cpu.size // len(self.layers)])
        logger.info("PLE pageable: prefetch ids match GPU ids: %s (tokens=%d reqs=%d)", ok, num_tokens, num_reqs)

    def _check_rows(self, num_reqs: int, num_tokens: int) -> None:
        # Void check (review 2026-09-23): in the serving process, on steps that mix a prefill with decodes,
        # the kernel's rows for the step's real ids must equal the bytes in the checkpoint files.
        ids_src, qsl_src, ctx_src = self.sources
        qsl = qsl_src[: num_reqs + 1].cpu().numpy()
        lens = np.diff(qsl)
        if lens.max(initial=0) <= 8 or (lens > 0).sum() < 2:
            return   # not a mixed step
        self.row_checks_left -= 1
        m = self.layers[0][0]
        ids = m.compute_ngram_ids(ids_src[:num_tokens], qsl_src[: num_reqs + 1], ctx_src[:num_reqs])
        got = self.table.gather(ids).view(torch.uint8).reshape(-1, self.table.row_bytes).cpu().numpy()
        flat = ids.reshape(-1).cpu().numpy()
        sh = flat // self.table.rows_per_shard
        loc = flat - sh * self.table.rows_per_shard
        ref = np.stack([self.table.views[int(a)][int(b)] for a, b in zip(sh, loc)])
        logger.info("PLE pageable: gathered rows match file bytes: %s (mixed step, reqs=%d, query_lens max=%d, "
                    "rows=%d)", bool(np.array_equal(got, ref)), int((lens > 0).sum()), int(lens.max()), flat.size)

    def _touch(self, rows: np.ndarray) -> None:
        h = self.host
        rows = np.sort(rows)
        if rows.size <= 4096:
            b = rows * self.row_bytes
            for p in np.unique(np.concatenate([b // _PAGE, (b + self.row_bytes - 1) // _PAGE])):
                os.posix_fadvise(self.fd, int(p) * _PAGE, _PAGE, os.POSIX_FADV_WILLNEED)
            int(h[rows, 0].sum()) + int(h[rows, self.row_bytes - 1].sum())
            return
        self.stats["big"] += 1
        chunks = np.array_split(rows, 64)
        list(self.pool.map(lambda c: int(h[c, 0].sum()) + int(h[c, -1].sum()), chunks))

    def _loop(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            slot, ev, num_reqs, num_tokens, t_sub = self.work.get()
            try:
                ev.synchronize()
                s = self.slots[slot]
                ids = s["ids"][:num_tokens].clone()
                qsl = s["qsl"][: num_reqs + 1].clone()
                ctx = s["ctx"][:num_reqs].clone()
            finally:
                self.free.put(slot)
            t0 = time.perf_counter()
            try:
                rows = self._cpu_rows(ids, qsl, ctx)
                rows = rows[(rows >= 0) & (rows < self.num_rows)]
                if self.table is not None:
                    if rows.size > 4096:
                        self.stats["big"] += 1
                    self.table.touch(rows, self.pool)
                else:
                    self._touch(rows)
            except Exception:
                logger.exception("PLE pageable: prefetch failed (gather unaffected)")
                continue
            t1 = time.perf_counter()
            st = self.stats
            st["steps"] += 1
            st["rows"] += int(rows.size)
            st["ms"] += (t1 - t0) * 1e3
            st["lag_ms"] += (t1 - t_sub) * 1e3
            if st["steps"] % 500 == 0 or num_tokens > 1024:
                logger.info("PLE pageable: steps=%d skipped=%d big=%d avg_rows=%.0f avg_prefetch_ms=%.2f "
                            "avg_lag_ms=%.2f last(tokens=%d rows=%d ms=%.1f)",
                            st["steps"], st["skipped"], st["big"], st["rows"] / st["steps"],
                            st["ms"] / st["steps"], st["lag_ms"] / st["steps"], num_tokens, rows.size,
                            (t1 - t0) * 1e3)


# ---------------------------------------------------------------------------------------------------------------
# v2: map the checkpoint's own safetensors files (no copy step). VLLM_PLE_PAGEABLE=checkpoint.
# The 128 shard tensors live in 10 files, in string order, starting at unaligned offsets (data starts at byte
# 2,239), so the GPU gathers through a 128-entry table of shard base addresses with byte loads.
# ---------------------------------------------------------------------------------------------------------------
import glob as _glob
import json as _json
import re as _re
import struct as _struct

import triton
import triton.language as tl

_SHARD_RE = _re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")


def pageable_mode() -> str:
    return os.environ.get("VLLM_PLE_PAGEABLE", "")


def is_pageable_shard_name(name: str) -> bool:
    return pageable_mode() == "checkpoint" and _SHARD_RE.search(name) is not None


@triton.jit
def _gather_rows_kernel(ids_ptr, shard_base_ptr, out_ptr, rows_per_shard, num_rows,
                        ROW: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.load(ids_ptr + pid).to(tl.int64)
    ok = (r >= 0) & (r < num_rows)
    r = tl.where(ok, r, 0)
    s = r // rows_per_shard
    local = r - s * rows_per_shard
    base = tl.load(shard_base_ptr + s).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, BLOCK)
    in_row = offs < ROW
    # Load only from valid rows; store the full row always, so an invalid id yields zeros and never
    # leaves allocator residue or a previous graph replay's bytes in the output.
    v = tl.load(base + local * ROW + offs, mask=in_row & ok, other=0)
    tl.store(out_ptr + pid.to(tl.int64) * ROW + offs, v, mask=in_row)


class CheckpointTable:
    """Read-only mappings of the checkpoint files that hold the PLE shards, plus a GPU gather over them."""

    def __init__(self, model_dir: str, num_rows: int, row_bytes: int, split_parts: int) -> None:
        require_pageable_access(torch.cuda.current_device())
        self.row_bytes = row_bytes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        shards: dict[int, tuple[str, int, list]] = {}
        files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
        for f in files:
            with open(f, "rb") as b:
                n = _struct.unpack("<Q", b.read(8))[0]
                h = _json.loads(b.read(n))
            for k, v in h.items():
                m = _SHARD_RE.search(k)
                if m and v.get("dtype") == "F8_E4M3":
                    i = int(m.group(1))
                    if i in shards:
                        raise ValueError(f"PLE shard {i} appears twice ({shards[i][0]}, {f})")
                    shards[i] = (f, 8 + n + v["data_offsets"][0], v["shape"])
        # Mirror Qwen4ExpNGramEmbedding.load_weights: shard i holds rows [i*S, min((i+1)*S, num_rows)),
        # S = ceil(num_rows / split_parts). Every shard that owns rows must exist with exactly that shape;
        # nothing is truncated, so every id < num_rows resolves inside its shard.
        shard_size = (num_rows + split_parts - 1) // split_parts
        needed = [i for i in range(split_parts) if i * shard_size < num_rows]
        missing = [i for i in needed if i not in shards]
        extra = sorted(set(shards) - set(needed))
        if missing or extra:
            raise ValueError(f"PLE pageable: {model_dir} shard set does not cover {num_rows} rows in "
                             f"{split_parts} parts: missing {missing[:8]}, unexpected {extra[:8]}")
        for i in needed:
            want = [min(shard_size, num_rows - i * shard_size), row_bytes]
            if list(shards[i][2]) != want:
                raise ValueError(f"PLE pageable: shard {i} has shape {shards[i][2]}, expected {want}")
        self.num_shards = len(needed)
        self.rows_per_shard = shard_size
        self.files: dict[str, dict] = {}
        for f in sorted({s[0] for s in shards.values()}):
            fd = os.open(f, os.O_RDONLY)
            size = os.path.getsize(f)
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
            arr = np.frombuffer(mm, dtype=np.uint8)
            addr = arr.__array_interface__["data"][0]
            if libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), _MADV_RANDOM) != 0:
                logger.warning("PLE pageable: madvise(MADV_RANDOM) failed on %s", f)
            _KEEP.extend([mm, arr])
            self.files[f] = dict(fd=fd, arr=arr, addr=addr, idx=len(self.files))
        fl = list(self.files)
        self.shard_file = np.array([self.files[shards[i][0]]["idx"] for i in range(self.num_shards)], np.int64)
        self.shard_off = np.array([shards[i][1] for i in range(self.num_shards)], np.int64)
        self.shard_rows = np.array([shards[i][2][0] for i in range(self.num_shards)], np.int64)
        self.file_list = [self.files[f] for f in fl]
        # Per-shard (rows x row_bytes) views for the CPU prefetch. Fancy-indexing these 2-D views, like v1's
        # contiguous table, is 15x faster cold than indexing the flat 5 GB byte arrays (touch_bench.py:
        # 51,200 rows 265-285 ms vs 4,080-4,117 ms; v1 240-243 ms).
        self.views = [self.files[shards[i][0]]["arr"][shards[i][1]:shards[i][1] + int(self.shard_rows[i]) * row_bytes]
                      .reshape(-1, row_bytes) for i in range(self.num_shards)]
        bases = [self.files[shards[i][0]]["addr"] + shards[i][1] for i in range(self.num_shards)]
        dev = torch.cuda.current_device()
        self.shard_base = torch.tensor(bases, dtype=torch.int64, device=f"cuda:{dev}")
        assert int(self.shard_rows.sum()) == num_rows
        self.num_rows = num_rows
        logger.info("PLE pageable: mapped %d shards x %d rows from %d checkpoint files in %s, no copy, nothing "
                    "pinned; GPU gathers through a %d-entry shard-address table", self.num_shards,
                    self.rows_per_shard, len(self.files), model_dir, self.num_shards)

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        out = torch.empty((flat.numel(), self.row_bytes), dtype=torch.uint8, device=flat.device)
        if flat.numel():
            _gather_rows_kernel[(flat.numel(),)](flat, self.shard_base, out, self.rows_per_shard,
                                                 self.num_rows, ROW=self.row_bytes,
                                                 BLOCK=triton.next_power_of_2(self.row_bytes))
        return out.view(torch.float8_e4m3fn).reshape(*ids.shape, self.row_bytes)

    # --- CPU side, for the prefetcher -------------------------------------------------------------------------
    def byte_addresses(self, rows: np.ndarray):
        s = np.clip(rows // self.rows_per_shard, 0, self.num_shards - 1)
        local = rows - s * self.rows_per_shard
        return self.shard_file[s], self.shard_off[s] + local * self.row_bytes

    def touch(self, rows: np.ndarray, pool) -> None:
        """Fault in both ends of every row (a 160 B row straddles a page ~4 % of the time)."""
        r = np.sort(rows)
        sh = r // self.rows_per_shard
        loc = r - sh * self.rows_per_shard
        last = self.row_bytes - 1

        def groups(idx):
            cs = sh[idx]
            cuts = np.flatnonzero(np.diff(cs)) + 1
            return [(self.views[int(g[0])], l) for g, l in zip(np.split(cs, cuts), np.split(loc[idx], cuts)) if g.size]

        if r.size <= 4096:
            fidx, off = self.byte_addresses(r)
            for fi in np.unique(fidx):
                o = off[fidx == fi]
                fd = self.file_list[int(fi)]["fd"]
                for p in np.unique(np.concatenate([o // _PAGE, (o + last) // _PAGE])):
                    os.posix_fadvise(fd, int(p) * _PAGE, _PAGE, os.POSIX_FADV_WILLNEED)
            for v, l in groups(np.arange(r.size)):
                int(v[l, 0].sum()) + int(v[l, last].sum())
            return
        tasks = []
        for ci in np.array_split(np.arange(r.size), 64):
            if ci.size:
                tasks.extend(groups(ci))
        list(pool.map(lambda t: int(t[0][t[1], 0].sum()) + int(t[0][t[1], last].sum()), tasks))


TABLE: dict = {}


def map_checkpoint_table(num_rows: int, row_bytes: int) -> "CheckpointTable":
    from vllm.config import get_current_vllm_config
    mc = get_current_vllm_config().model_config
    split = int(getattr(mc.hf_text_config, "split_ngram_parts", 512))
    t = CheckpointTable(mc.model, num_rows, row_bytes, split)
    TABLE["t"] = t
    return t
