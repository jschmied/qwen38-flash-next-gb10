"""Does persistent_topk (stock _C, and our deterministic _C_det) ever place -1 before a real id? QSA-style calls:
per-row visible lengths from 1 to 3*top_k (underfull rows included), random and tied logits. Prints counts."""
import sys, torch
import vllm._custom_ops  # noqa: F401  registers torch.ops._C
det = len(sys.argv) > 1
if det:
    torch.ops.load_library("/opt/llm/kernel-det/_C_det.so")
op = torch.ops._C_det.persistent_topk if det else torch.ops._C.persistent_topk
ws = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
g = torch.Generator(device="cuda").manual_seed(0)
tot = bad = underfull = 0
for top_k in (512, 1024, 2048):
    for max_len in (top_k // 2, top_k, 3 * top_k):
        rows = 64
        for kind in ("rand", "ties"):
            logits = torch.randn(rows, max_len, device="cuda", generator=g)
            if kind == "ties":
                logits = logits.round()
            lengths = torch.randint(1, max_len + 1, (rows,), device="cuda", generator=g, dtype=torch.int32)
            out = torch.empty(rows, top_k, dtype=torch.int32, device="cuda")
            op(logits, lengths, out, ws, top_k, max_len)
            torch.cuda.synchronize()
            neg = out < 0
            # a -1 before a real id: some position j with out[j] < 0 and a later out[j'] >= 0
            later_real = torch.flip(torch.cummax(torch.flip((~neg).int(), [1]), 1).values, [1])
            interleaved = (neg[:, :-1] & (later_real[:, 1:] > 0)).any(dim=1)
            tot += rows; bad += int(interleaved.sum()); underfull += int((lengths < top_k).sum())
print({"kernel": "_C_det" if det else "_C (stock)", "rows": tot, "underfull_rows": underfull,
       "rows_with_minus1_before_real_id": bad})
# order of the real ids: already ascending?
asc = unsorted = 0
g = torch.Generator(device="cuda").manual_seed(1)
for top_k in (512, 1024, 2048):
    for max_len in (top_k // 2, top_k, 3 * top_k):
        logits = torch.randn(64, max_len, device="cuda", generator=g)
        lengths = torch.randint(1, max_len + 1, (64,), device="cuda", generator=g, dtype=torch.int32)
        out = torch.empty(64, top_k, dtype=torch.int32, device="cuda")
        op(logits, lengths, out, ws, top_k, max_len); torch.cuda.synchronize()
        for r in range(64):
            v = out[r][out[r] >= 0]
            if torch.equal(v, torch.sort(v).values): asc += 1
            else: unsorted += 1
print({"rows_real_ids_ascending": asc, "rows_real_ids_not_ascending": unsorted})
