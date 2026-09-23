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
    if not getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "is_integrated", 1):
        raise RuntimeError("VLLM_PLE_PAGEABLE_FILE needs an integrated GPU with pageable memory access")
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
        self.host = MAPPING["host"]
        self.fd = MAPPING["fd"]
        self.row_bytes = MAPPING["row_bytes"]
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
                rows = rows[(rows >= 0) & (rows < self.host.shape[0])]
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
