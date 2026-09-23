import sys, time, json, numpy as np, torch
sys.path.insert(0, "/opt/llm/runtime/vllm-venv-fnmain3/lib/python3.12/site-packages")
from vllm.v1.ple_offload.pageable import CheckpointTable
t = CheckpointTable("/opt/llm/models/qwen38-flash-next-mtpfp4", 320001536, 160)
ref = np.memmap("/opt/llm/models/ple-cache/qwen38-fn-plefp8.bin", dtype=np.uint8, mode="r").reshape(-1, 160)
S = t.rows_per_shard; rng = np.random.default_rng(99)
edges = np.concatenate([np.arange(t.num_shards) * S, np.arange(t.num_shards) * S + S - 1])
ids = np.concatenate([edges, rng.integers(0, t.num_rows, 500_000)]).astype(np.int64)
res = {}
g = t.gather(torch.from_numpy(ids).cuda()); torch.cuda.synchronize()
res["k1_bitexact"] = bool(np.array_equal(g.view(torch.uint8).cpu().numpy(), ref[ids]))
res["k1_rows"] = int(ids.size)
x = torch.from_numpy(ids[:64].copy()).cuda()
eager = t.gather(x).view(torch.uint8).clone()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    t.gather(x)
torch.cuda.current_stream().wait_stream(st)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    y = t.gather(x)
x.copy_(torch.from_numpy(ids[64:128].copy()).cuda()); gr.replay(); torch.cuda.synchronize()
res["k2_graph_eq_eager_new_ids"] = bool(torch.equal(y.view(torch.uint8), t.gather(x).view(torch.uint8)))
x.copy_(torch.from_numpy(ids[:64].copy()).cuda()); gr.replay(); torch.cuda.synchronize()
res["k2_graph_eq_eager_orig"] = bool(torch.equal(y.view(torch.uint8), eager))
ts = []
for _ in range(50):
    torch.cuda.synchronize(); t0 = time.perf_counter(); t.gather(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
res["k3_warm64_us"] = [round(min(ts), 1), round(float(np.median(ts)), 1), round(max(ts), 1)]
print(json.dumps(res))
