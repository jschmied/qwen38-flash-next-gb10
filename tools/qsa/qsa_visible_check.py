#!/usr/bin/env python3
"""QSA decode-selection bound check: every returned block index < row's visible.

Run with:  /opt/llm/runtime/vllm-venv-rssm/bin/python qsa_visible_check.py
Needs one GPU briefly (a few hundred small kernel launches, < 100 MB).

What it exercises (the exact code the server runs for decode rows):
  vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.qsa_select_paged_decode
      -> _qsa_mqa_paged_uniform_kernel (logits, masked per row by visible)
      -> _topk(...) -> torch.ops._C.persistent_topk  (stock)
                    or torch.ops._C_det.persistent_topk (VLLM_QSA_DET_TOPK set)
The det switch is read by _topk from os.environ on EVERY call
(VLLM_QSA_DET_TOPK truthy; library path VLLM_QSA_DET_LIB, default
/opt/llm/kernel-det/_C_det.so, loaded once and cached on _topk._qsadet_loaded).
This script mirrors that by setting/unsetting the env vars around each call.

Two paths per case:
  topk  : call _topk directly with a logits tensor we build: columns < visible
          get finite random scores, columns >= visible get FILL (+inf or 1e30).
  e2e   : call qsa_select_paged_decode itself. Its logits buffer comes from
          torch.empty inside the module; we swap the module's `torch` for a
          shim whose empty() returns that buffer pre-filled with FILL, so every
          column the kernel does not write (>= visible for that row) holds FILL.
          The K cache row at column g is also made maximally aligned with q so
          rows with visible=g would score it highest if the per-row store mask
          leaked.
Rows: one request, decode_query_len=4, visible = [g, g, g, g+1].
"""

import inspect
import os
import sys
import traceback

import torch

EXPECTED_SELECT_PARAMS = [
    "q", "k_cache", "page_table", "visible_blocks", "token_topk",
    "compress_ratio", "decode_query_len", "block_indices",
]
EXPECTED_TOPK_PARAMS = [
    "logits", "visible_blocks", "token_topk", "compress_ratio",
    "block_indices", "topk_workspace",
]
EXPECTED_OP_SCHEMA_PREFIX = (
    "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
    "Tensor workspace, int k, int max_seq_len)"
)

# Model constants (Qwen3.8-Flash-Next config.json: indexer_budget 2048,
# indexer_compress_ratio 4, indexer_n_heads 4, indexer_head_dim 128).
COMPRESS_RATIO = 4
NUM_HEADS = 4
HEAD_DIM = 128
DECODE_QUERY_LEN = 4
PAGE_SIZE = 64                      # compressed rows per cache page
G_VALUES = [1, 5, 511, 512, 2047]
TOKEN_TOPKS = [64, 2048, 8192]      # block_topk 16 / 512 (prod) / 2048
COLUMN_WIDTHS = [4096, 65536]       # 65536 compressed rows = 256k-token context
FILLS = [("inf", float("inf")), ("1e30", 1e30)]
DET_LIB_DEFAULT = "/opt/llm/kernel-det/_C_det.so"


def die(msg: str) -> None:
    print(f"FATAL: {msg}", flush=True)
    sys.exit(2)


# --------------------------------------------------------------------------
# Imports and signature checks (fail loudly on any mismatch)
# --------------------------------------------------------------------------
if not torch.cuda.is_available():
    die("CUDA not available")

try:
    import vllm._custom_ops  # noqa: F401  loads _C_stable_libtorch, registers torch.ops._C.persistent_topk
except Exception as e:  # noqa: BLE001
    die(f"import vllm._C failed: {e!r}")

try:
    import vllm.models.qwen4_exp.nvidia.ops.qsa_indexer as qi
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    die(f"import qsa_indexer failed: {e!r}")

for name in ("qsa_select_paged_decode", "_topk", "_TOPK_WORKSPACE_BYTES"):
    if not hasattr(qi, name):
        die(f"qsa_indexer has no attribute {name!r}")

sel_sig = inspect.signature(qi.qsa_select_paged_decode)
topk_sig = inspect.signature(qi._topk)
if list(sel_sig.parameters) != EXPECTED_SELECT_PARAMS:
    die(f"qsa_select_paged_decode signature changed: {sel_sig}")
if list(topk_sig.parameters) != EXPECTED_TOPK_PARAMS:
    die(f"_topk signature changed: {topk_sig}")
print(f"CALL qsa_indexer.qsa_select_paged_decode{sel_sig}")
print(f"CALL qsa_indexer._topk{topk_sig}")

topk_src = inspect.getsource(qi._topk)
for needle in ('"VLLM_QSA_DET_TOPK"', '"VLLM_QSA_DET_LIB"', "_C_det.persistent_topk",
               "_C.persistent_topk", "_C.cooperative_topk"):
    if needle not in topk_src:
        die(f"_topk source no longer contains {needle}; det/stock dispatch changed")

if not hasattr(torch.ops._C, "persistent_topk"):
    die("torch.ops._C.persistent_topk not registered")
stock_schema = str(torch.ops._C.persistent_topk.default._schema)
import re as _re
_norm = lambda x: _re.sub(r"\(\$\d+!?\s*->\s*\)|!", "", x)  # alias annotations differ across torch versions
if not _norm(stock_schema).split("::")[-1].startswith(_norm(EXPECTED_OP_SCHEMA_PREFIX)):
    die(f"_C.persistent_topk schema changed: {stock_schema}")
print(f"OP   {stock_schema}")

# Mirror _topk's stock choice so the log says which op actually ran.
from vllm.platforms import current_platform  # noqa: E402


def stock_op_name(logits: torch.Tensor) -> str:
    coop = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    return "_C.cooperative_topk" if coop else "_C.persistent_topk"


det_lib = os.environ.get("VLLM_QSA_DET_LIB", DET_LIB_DEFAULT)
variants = ["stock"]
if os.path.exists(det_lib):
    variants.append("det")
else:
    print(f"SKIP det variant: {det_lib} not found")


class _Env:
    """Set the env exactly as the server's det switch reads it."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.saved = {}

    def __enter__(self):
        for k in ("VLLM_QSA_DET_TOPK", "VLLM_QSA_DET_LIB"):
            self.saved[k] = os.environ.pop(k, None)
        if self.variant == "det":
            os.environ["VLLM_QSA_DET_TOPK"] = "1"
            os.environ["VLLM_QSA_DET_LIB"] = det_lib
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        return False


class _TorchShim:
    """Stands in for the module's `torch`; pre-fills the float32 logits buffer."""

    def __init__(self, fill: float) -> None:
        self.fill = fill
        self.logits_allocs = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        t = torch.empty(*args, **kwargs)
        if t.dtype == torch.float32 and t.dim() == 2:
            t.fill_(self.fill)
            self.logits_allocs += 1
        return t


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_rows(out: torch.Tensor, visible: torch.Tensor, block_topk: int,
               ref_sets=None) -> list[str]:
    errs = []
    out = out.cpu()
    for r in range(out.shape[0]):
        v = int(visible[r])
        row = out[r].tolist()
        m = min(v, block_topk)
        bad = [x for x in row if x >= v or x < -1]
        if bad:
            errs.append(f"row{r} v={v}: index>=visible or <-1: {bad[:6]}")
        head = row[:m]  # expand_qsa_block_indices reads exactly these ranks
        if any(x < 0 or x >= v for x in head):
            errs.append(f"row{r} v={v}: invalid entry in first {m}: "
                        f"{[x for x in head if x < 0 or x >= v][:6]}")
        if len(set(head)) != len(head):
            errs.append(f"row{r} v={v}: duplicate indices in first {m}")
        if ref_sets is not None and set(head) != ref_sets[r]:
            errs.append(f"row{r} v={v}: selected set != reference top-{m}")
    return errs


def run_topk_direct(variant, g, token_topk, width, fill_val, gen):
    block_topk = token_topk // COMPRESS_RATIO
    visible = torch.tensor([g, g, g, g + 1], dtype=torch.int32, device="cuda")
    logits = torch.full((DECODE_QUERY_LEN, width), fill_val,
                        dtype=torch.float32, device="cuda")
    ref_sets = []
    for r in range(DECODE_QUERY_LEN):
        v = int(visible[r])
        # distinct finite non-negative scores (the kernel emits relu'd sums)
        vals = torch.randperm(v, generator=gen).to(torch.float32) + 1.0
        logits[r, :v] = vals.cuda()
        m = min(v, block_topk)
        ref_sets.append(set(torch.topk(vals, m).indices.tolist()))
    block_indices = torch.full((DECODE_QUERY_LEN, block_topk), -1,
                               dtype=torch.int32, device="cuda")
    ws = torch.empty((qi._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda")
    with _Env(variant):
        qi._topk(logits, visible, token_topk, COMPRESS_RATIO, block_indices, ws)
    torch.cuda.synchronize()
    return check_rows(block_indices, visible.cpu(), block_topk, ref_sets), logits


def run_e2e(variant, g, token_topk, width, fill_val, gen):
    block_topk = token_topk // COMPRESS_RATIO
    visible = torch.tensor([g, g, g, g + 1], dtype=torch.int32, device="cuda")
    num_table_pages = width // PAGE_SIZE
    used_pages = (g + 1 + PAGE_SIZE - 1) // PAGE_SIZE + 1
    # physical page 0 is a dummy; logical page i -> physical i+1 for used pages
    page_table = torch.zeros((1, num_table_pages), dtype=torch.int32, device="cuda")
    page_table[0, :used_pages] = torch.arange(1, used_pages + 1, dtype=torch.int32)
    k_cache = torch.randn((used_pages + 1, PAGE_SIZE, 1, HEAD_DIM),
                          generator=gen).abs().mul_(0.01).to(torch.bfloat16).cuda()
    q = torch.randn((DECODE_QUERY_LEN, NUM_HEADS, HEAD_DIM),
                    generator=gen).abs().to(torch.bfloat16).cuda()
    # Column g (visible only to row 3) gets a key aligned with every query:
    # if rows 0..2 leaked it, it would be their top score.
    kg = q.float().mean(dim=(0, 1)) * 100.0
    phys, off = divmod(g, PAGE_SIZE)
    k_cache[phys + 1, off, 0] = kg.to(torch.bfloat16)
    block_indices = torch.full((DECODE_QUERY_LEN, block_topk), -1,
                               dtype=torch.int32, device="cuda")
    shim = _TorchShim(fill_val)
    real_torch = qi.torch
    qi.torch = shim
    try:
        with _Env(variant):
            qi.qsa_select_paged_decode(
                q, k_cache, page_table, visible, token_topk,
                COMPRESS_RATIO, DECODE_QUERY_LEN, block_indices,
            )
        torch.cuda.synchronize()
    finally:
        qi.torch = real_torch
    if shim.logits_allocs != 1:
        die(f"expected 1 float32 2D logits alloc inside qsa_select_paged_decode, "
            f"saw {shim.logits_allocs}; allocation pattern changed, prefill invalid")
    errs = check_rows(block_indices, visible.cpu(), block_topk)
    b = block_indices.cpu()
    # row 3 sees column g; with the aligned key it is the top score there
    if g not in b[3].tolist()[: min(g + 1, block_topk)]:
        errs.append(f"row3 v={g+1}: aligned column {g} not selected "
                    f"(logits kernel/topk suspect)")
    return errs


def main() -> int:
    gen = torch.Generator().manual_seed(1234)
    n_fail = 0
    n_total = 0
    print(f"device={torch.cuda.get_device_name()} "
          f"cap={torch.cuda.get_device_capability()} variants={variants}")
    for variant in variants:
        for width in COLUMN_WIDTHS:
            probe = torch.empty((DECODE_QUERY_LEN, width), dtype=torch.float32,
                                device="cuda")
            op = "_C_det.persistent_topk" if variant == "det" else stock_op_name(probe)
            del probe
            for g in G_VALUES:
                if g + 1 > width:
                    continue
                for token_topk in TOKEN_TOPKS:
                    block_topk = token_topk // COMPRESS_RATIO
                    rel = "k<g" if block_topk < g else ("k==g" if block_topk == g else "k>g")
                    for fill_name, fill_val in FILLS:
                        for path in ("topk", "e2e"):
                            n_total += 1
                            tag = (f"{variant:5s} op={op} path={path:4s} width={width} "
                                   f"g={g} vis=[{g},{g},{g},{g+1}] "
                                   f"block_topk={block_topk}({rel}) fill={fill_name}")
                            try:
                                if path == "topk":
                                    errs, _ = run_topk_direct(variant, g, token_topk,
                                                              width, fill_val, gen)
                                else:
                                    errs = run_e2e(variant, g, token_topk,
                                                   width, fill_val, gen)
                            except SystemExit:
                                raise
                            except Exception as e:  # noqa: BLE001
                                n_fail += 1
                                print(f"ERROR {tag}: {type(e).__name__}: {e}",
                                      flush=True)
                                continue
                            if errs:
                                n_fail += 1
                                print(f"FAIL  {tag}: {'; '.join(errs[:3])}", flush=True)
                            else:
                                print(f"PASS  {tag}", flush=True)
        if variant == "det" and not getattr(qi._topk, "_qsadet_loaded", False):
            die("det variant ran but _topk._qsadet_loaded is not set; "
                "the det library was never loaded")
    print(f"SUMMARY {n_total - n_fail}/{n_total} passed, {n_fail} failed/errored")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
