"""Gate for the NVFP4 draft head, on the REAL 32k slice of mtpfp4's FP8 lm_head (prod down).
1. kernel vs torch reference (same quantized weights): must agree to fp32-accumulation noise, argmax 100 %
2. quantization effect vs today's BF16 slice: logit rel-RMSE and argmax agreement on random hidden states
   (informational only -- random x is not the drafter's distribution; acceptance on the server is the real test)
3. CUDA-graph timing, M in {1,2,4,8,16}: BF16 F.linear (today) vs NVFP4 kernel over a small config sweep
Writes best config to argv[2] as 'block_n,num_warps'. Exit 2 (VOID) on a correctness failure, 3 if not faster."""
import importlib.util, json, statistics, sys
import torch
from safetensors import safe_open
D = "/opt/llm/models/qwen38-flash-next-mtpfp4"
sp = importlib.util.spec_from_file_location("h", sys.argv[1]); h = importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
m = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{D}/{m['lm_head.weight']}", "pt", device="cuda") as f:
    W = f.get_tensor("lm_head.weight"); S = f.get_tensor("lm_head.weight_scale_inv")
ids = sorted({int(x) for x in open("/opt/llm/runners/dv/draft_vocab_32768.txt").read().split() if x.strip()})
idx = torch.tensor(ids, device="cuda")
rows = W.index_select(0, idx).float(); srows = S.index_select(0, idx // 128).float()
wf = rows * srows.repeat_interleave(128, dim=1)
bf = wf.to(torch.bfloat16).contiguous()                       # today's slice, exactly as FNDV builds it
q, s, g = h.quantize_nvfp4_rows(wf)
ref = h.dequantize_nvfp4_rows(q, s, g)
print(f"slice {tuple(bf.shape)}: bf16 {bf.numel()*2/2**20:.1f} MiB -> nvfp4 {(q.numel()+s.numel())/2**20:.1f} MiB, g={g:.3e}", flush=True)
print(f"weight rel-RMSE nvfp4 vs exact: {float((ref-wf).pow(2).mean().sqrt()/wf.pow(2).mean().sqrt()):.4f}", flush=True)
torch.manual_seed(0)
bad = []
for M in (1, 4, 16, 37):
    x = (torch.randn(M, wf.shape[1], device="cuda") * 1.0).to(torch.bfloat16)
    out = h.nvfp4_rows_gemv(x, q, s, g)
    r = x.float() @ ref.T
    rel = float((out - r).abs().max() / r.abs().max())
    am = float(out.argmax(1).eq(r.argmax(1)).float().mean())
    lb = (x @ bf.T).float()
    qrel = float((r - lb).pow(2).mean().sqrt() / lb.pow(2).mean().sqrt())
    qam = float(r.argmax(1).eq(lb.argmax(1)).float().mean())
    print(f"M={M:2d} kernel-vs-ref max rel {rel:.2e} argmax {am:.3f} | quant effect: logit rel-RMSE {qrel:.4f}, argmax agree vs bf16 {qam:.3f}", flush=True)
    if rel > 1e-3 or am < 1.0:
        bad.append(M)
if bad:
    print(f"VOID: kernel disagrees with reference at M={bad}"); sys.exit(2)


def gtime(fn, reps=20):
    fn(); torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st):
        fn()
        with torch.cuda.graph(gr, stream=st):
            for _ in range(reps): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3): gr.replay()
        b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000 / (3 * reps))
    return statistics.median(ts)


cfgs = [(32, 4), (64, 4), (64, 8), (128, 4), (128, 8), (256, 8)]
best = None; res = {}
for M in (1, 2, 4, 8, 16):
    x = torch.randn(M, wf.shape[1], device="cuda").to(torch.bfloat16)
    tb = gtime(lambda: torch.nn.functional.linear(x, bf))
    row = {"bf16": tb}
    for bn, nw in cfgs:
        try:
            row[f"{bn},{nw}"] = gtime(lambda: h.nvfp4_rows_gemv(x, q, s, g, block_n=bn, num_warps=nw))
        except Exception as ex:
            row[f"{bn},{nw}"] = float("nan"); print(f"  cfg {bn},{nw} failed: {type(ex).__name__}")
    res[M] = row
    print(f"M={M:2d} " + "  ".join(f"{k}:{v:7.1f}us" for k, v in row.items()), flush=True)
import math
score = {c: statistics.mean(res[M][f"{c[0]},{c[1]}"] for M in (1, 2, 4)) for c in cfgs}
score = {c: (v if not math.isnan(v) else math.inf) for c, v in score.items()}
bc = min(score, key=score.get)
ratio = statistics.mean(res[M][f"{bc[0]},{bc[1]}"] / res[M]["bf16"] for M in (1, 2, 4))
print(f"best cfg block_n={bc[0]} num_warps={bc[1]}; nvfp4/bf16 time at M=1,2,4: {ratio:.3f}", flush=True)
open(sys.argv[2], "w").write(f"{bc[0]},{bc[1]}\n")
if ratio > 0.9:
    print("NOT FASTER: skip the server A/B"); sys.exit(3)
print("== HEADBENCH OK ==")
