#!/usr/bin/env python3
"""Boundary/off-by-one test for persistent_topk. Correctness, not determinism.

Contract (from the torch fallback in qsa.py): valid indices are [0, visible_blocks). Everything
at or beyond visible_blocks must be ignored.

An off-by-one has two very different signatures, and only one is visible to a determinism test:

  misses index 0 or visible-1   deterministic and silently WRONG. Every run agrees, so replay,
                                rate tests and A/B comparisons all pass while the model quietly
                                degrades. Only an oracle catches it.
  reads index visible (or more) reads undefined memory -> wrong AND nondeterministic, and the
                                magnitude is unbounded because the bytes are arbitrary.

Tests, each with a known-correct answer, so no captured input is needed:
  T1 sentinel  logits[:, visible:] = +inf. A correct kernel ignores them. Any returned index
               >= visible is a read past the boundary, caught directly rather than inferred.
  T2 first     the maximum placed at index 0            -> 0 must be selected
  T3 last      the maximum placed at index visible-1    -> visible-1 must be selected
  T4 oracle    vs torch.topk on the masked logits, compared as SETS (tie order may legitimately
               differ; membership may not)
  T5 sweep     visible swept over off-by-one-prone values, incl. 4096 and 8448 where a public
               report puts a value error of 1.2e-2

Usage: python topk_boundary.py [rows] [k]
"""
import sys, torch

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16

import vllm  # noqa: F401
from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as _qsa

WS = getattr(_qsa, "_TOPK_WORKSPACE_BYTES", 1 << 22)
op = torch.ops._C.persistent_topk
dev = "cuda"
print(f"  device {torch.cuda.get_device_name(0)}  rows={rows} k={K} workspace={WS}B")
print(f"  op: persistent_topk (sm_121 takes this path; cooperative_topk is gated off)\n")


def call(logits, visible, columns):
    blocks = torch.zeros((logits.shape[0], K), dtype=torch.int32, device=dev)
    ws = torch.empty((WS,), dtype=torch.uint8, device=dev)
    op(logits, visible, blocks, ws, K, columns)
    torch.cuda.synchronize()
    return blocks


def oracle(logits, visible):
    lg = logits.clone()
    ar = torch.arange(lg.shape[1], device=dev).view(1, -1)
    lg.masked_fill_(ar >= visible.to(torch.long).view(-1, 1), float("-inf"))
    return torch.topk(lg, K, dim=1, sorted=True).indices


fails = []
COLS = 16384

for vis in [K, K + 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
            1023, 1024, 1025, 4095, 4096, 4097, 8447, 8448, 8449]:
    if vis > COLS or vis < K:
        continue
    visible = torch.full((rows,), vis, dtype=torch.int32, device=dev)

    # T1 sentinel: +inf beyond the boundary. Any index >= vis means it read too far.
    lg = torch.randn(rows, COLS, dtype=torch.float32, device=dev)
    lg[:, vis:] = float("inf")
    b = call(lg, visible, COLS)
    over = (b >= vis).sum().item()
    if over:
        fails.append(f"vis={vis}: T1 returned {over} index(es) >= visible -> READS PAST BOUNDARY")

    # T2 first element
    lg = torch.full((rows, COLS), -1.0, dtype=torch.float32, device=dev)
    lg[:, 0] = 100.0
    lg[:, vis:] = float("inf")
    b = call(lg, visible, COLS)
    if not (b == 0).any(dim=1).all().item():
        fails.append(f"vis={vis}: T2 index 0 NOT selected though it holds the maximum")

    # T3 last valid element
    lg = torch.full((rows, COLS), -1.0, dtype=torch.float32, device=dev)
    lg[:, vis - 1] = 100.0
    lg[:, vis:] = float("inf")
    b = call(lg, visible, COLS)
    if not (b == vis - 1).any(dim=1).all().item():
        fails.append(f"vis={vis}: T3 last valid index {vis-1} NOT selected though it holds the maximum")

    # T4 oracle, as sets
    lg = torch.randn(rows, COLS, dtype=torch.float32, device=dev)
    b = call(lg, visible, COLS)
    o = oracle(lg, visible)
    for r in range(rows):
        if set(b[r].tolist()) != set(o[r].tolist()):
            miss = set(o[r].tolist()) - set(b[r].tolist())
            extra = set(b[r].tolist()) - set(o[r].tolist())
            fails.append(f"vis={vis} row{r}: T4 disagrees with torch.topk "
                         f"missing={sorted(miss)[:4]} extra={sorted(extra)[:4]}")
            break

# T5/T6: the UNDERFILLED case (visible < k), which is our 60-token probe (~15 visible, k=512).
# T5: what does persistent_topk WRITE into slots >= visible? Pre-fill `blocks` with a sentinel;
#     untouched sentinel = kernel leaves them (so they hold torch.empty garbage in production).
# T6: the blocks.sort() hook added on 2026-09-01 (vllm#54521 "determinism fix", NOT in the stock
#     file) is applied to the whole row. If padding is garbage, sorting can move garbage BELOW the
#     real indices, and the expansion kernel then reads garbage from slots 0..visible-1.
#     This would make the '#' vs 'The' top-token difference OUR bug, not the kernel's.
print("  === underfilled case: visible < k (the probe's regime) ===")
for vis in (15, 60, 200):
    visible = torch.full((rows,), vis, dtype=torch.int32, device=dev)
    lg = torch.randn(rows, COLS, dtype=torch.float32, device=dev)
    lg[:, vis:] = float("inf")
    SENT = 2_000_000_000
    blocks = torch.full((rows, K), SENT, dtype=torch.int32, device=dev)
    ws = torch.empty((WS,), dtype=torch.uint8, device=dev)
    op(lg, visible, blocks, ws, K, COLS); torch.cuda.synchronize()
    pad = blocks[:, vis:]
    untouched = int((pad == SENT).sum()); total = pad.numel()
    valid_ok = bool(((blocks[:, :vis] >= 0) & (blocks[:, :vis] < vis)).all())
    print(f"    vis={vis:>3} k={K}: valid slots in-range={valid_ok}  padding untouched={untouched}/{total}"
          + ("  <- kernel LEAVES padding; production holds torch.empty garbage there" if untouched == total
             else f"  padding written with e.g. {sorted(set(pad[0].tolist()))[:4]}"))
    # T6: simulate production: garbage padding + the sort hook
    garbage = blocks.clone(); garbage[:, vis:] = torch.randint(-5, 3, (rows, K - vis), device=dev, dtype=torch.int32)
    sorted_ = garbage.sort(dim=1).values
    head_ok = bool(((sorted_[:, :vis] >= 0) & (sorted_[:, :vis] < vis)).all())
    real = set(range(vis)); got = set(sorted_[0, :vis].tolist())
    print(f"           with garbage padding + .sort(): first {vis} slots still the real set = {got == real}"
          + ("" if got == real else f"  <- SORT MIXES GARBAGE IN: {sorted(got - real)[:5]} replaced {sorted(real - got)[:5]}"))
    if not head_ok or got != real:
        fails.append(f"vis={vis}: T6 the blocks.sort() hook corrupts the valid range when padding is garbage")

print("  === results ===")
if not fails:
    print("  all boundary tests PASS: no off-by-one at either end, no read past visible_blocks,")
    print("  and selection matches torch.topk as a set at every swept size.")
else:
    for f in fails:
        print("  FAIL " + f)
print(f"\n  {len(fails)} failure(s)")
print("\n  Note: T2/T3 failing is a SILENT quality bug -- deterministic, so every replay and")
print("  A/B test we run would agree with itself while the model degrades. T1 failing is the")
print("  nondeterministic variant and would explain large logit jumps.")
