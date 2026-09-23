# I1 offline verifier: for every recorded step, ids == CPU hash of the recorded inputs, rows == checkpoint bytes.
import sys, json, numpy as np, torch
from types import SimpleNamespace
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
rec = torch.load(sys.argv[1], weights_only=False)
ns = SimpleNamespace(layer_multipliers=rec["layer_multipliers"], ngram_heads_vocab_sizes=rec["ngram_heads_vocab_sizes"],
                     ngram_heads_offsets=rec["ngram_heads_offsets"], eos_token_id=rec["eos_token_id"],
                     ngram_size=rec["ngram_size"], heads_per_ngram=rec["heads_per_ngram"],
                     _shift_precompute=Qwen4ExpNGramEmbedding._shift_precompute, _shift_apply=Qwen4ExpNGramEmbedding._shift_apply)
lay = rec["layout"]; R = rec["row_bytes"]
views = [np.memmap(s.path, dtype=np.uint8, mode="r", offset=s.offset, shape=(s.rows, R)) for s in lay.shards]
res = dict(steps=0, mixed=0, ids_ok=0, rows_ok=0, mixed_ids_ok=0, mixed_rows_ok=0, rows_checked=0)
for st in rec["steps"]:
    ids = st["ids"]; ntok = ids.shape[0]
    cpu = Qwen4ExpNGramEmbedding.compute_ngram_ids(ns, st["input_ids"].long(), st["qsl"].long(), st["ctx"].long())
    ids_ok = torch.equal(cpu.to(ids.dtype), ids)
    flat = ids.reshape(-1).numpy().astype(np.int64)
    ref = np.stack([views[r // lay.rows_per_shard][r % lay.rows_per_shard] for r in flat])
    rows_ok = np.array_equal(st["rows"].reshape(-1, R).numpy(), ref)
    lens = np.diff(st["qsl"].numpy()); mixed = bool((lens > 8).any() and ((lens >= 1) & (lens <= 4)).any())
    res["steps"] += 1; res["ids_ok"] += ids_ok; res["rows_ok"] += rows_ok; res["rows_checked"] += flat.size
    if mixed: res["mixed"] += 1; res["mixed_ids_ok"] += ids_ok; res["mixed_rows_ok"] += rows_ok
print(json.dumps(res))
