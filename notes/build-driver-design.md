# Build driver — design

Revised **2026-09-10** after the index census. The first version's chained loop rested on a wrong
fact; the census replaced it with a cheaper shape.

## What the source actually looks like

Censused from `model.safetensors.index.json` on the PBS copy
(`10.0.0.70:/mnt/bulk/hf/Qwen--Qwen3.8-Flash-Next`, 1,658 tensors, 131 shards, **335.3 GiB**):

| component | size | share |
| --- | --- | --- |
| experts | 229.7 GiB | 68.5 % |
| ple | 95.4 GiB | 28.5 % |
| layer-dense | 6.9 GiB | 2.1 % |
| embed | 1.2 GiB | 0.4 % |
| lm_head | 1.2 GiB | 0.4 % |
| other | 0.9 GiB | 0.3 % |

**Correction to the first draft:** it said "layer 0 spans shards 1..112, so fetch by tensor, not by
shard". The conclusion was right, the reason was not. A layer touches a **median of 3 shards**, and
each layer's experts are exactly two fused tensors — `mlp.experts.gate_up_proj` (512, 1280, 2560) and
`mlp.experts.down_proj` (512, 2560, 640), **4.68 GiB together, in 2 shards**. The 112-shard figure
came from layer 1, which is the outlier: **all 137 PLE tensors are named `layers.1.ple.*`**, so that
one layer weighs 100.26 GiB against every other layer's 4.83. Fetch by tensor anyway — layers sit in
arbitrary shard order (layer 5 → shard 121, layer 24 → shard 69), so shard-order streaming buys
nothing.

## The revised shape: capture once, then calibrate layers independently

The chained loop (run layer i, capture its output as layer i+1's input) forced a strict order and
needed every component *including the 95 GiB PLE* resident to compute the forward. It is unnecessary.

**Stage 1 — capture (GPU, one prefill pass).** Run a token budget of the corpus through the *served*
model, hooking each layer's `Qwen3NextSparseMoeBlock.forward`, and write per layer the MoE input rows
plus the router's top-k ids.

**Stage 2 — calibrate (per layer, independent).** For layer L: stream its 4.68 GiB of BF16 experts
from PBS by byte range, load layer L's captured rows, calibrate, export, drop.

**Stage 3 — assemble.** `reshard.py`, sha256 per tensor, then serve.

What this buys: no layer ordering, no PLE, no 335 GiB resident anywhere, resumable at layer
granularity, and layers may run out of order or in parallel. What it costs: the activations come
from the current *quantized* model rather than BF16 — a second-order error on the Hessian's input
distribution, and unavoidable regardless, since BF16 Flash-Next does not fit in 128 GB.

## Streaming cost, measured

PBS link is 1 GbE, measured **91.6 MB/s** GB10←PBS. Experts + dense = 236.6 GiB ≈ **43 min** for a
whole-model pass, ~51 s per layer, overlapped with that layer's calibration.

## The three things the driver must do (unchanged, all now in code)

1. **Feed `down_proj` from the intermediate, not the layer input.** Its input is the expert's own
   intermediate activation. The driver computes it in BF16 in the loop —
   `inter = silu(a) * b` over `chunk(gate_up(x), 2)` — rather than capturing it. Calibrating
   `down_proj` from the layer input would silently produce a half-max-calibrated model.
2. **Count and report LH / thin / no-rows per layer.** On one layer with an 8,192-row cap: 297
   experts got ≥64 routed rows, 194 got 1–63, and **21 got none** and fell back to weight-only max —
   a fallback that is invisible in the output checkpoint. The driver emits the three counts per layer
   and **fails the layer** above `MAX_NOROWS = 8` instead of exporting quietly.
3. **Verify by iteration count, never by "Calibration complete."** `NVFP4_DEFAULT_CFG` with
   `algorithm=local_hessian` calibrates **zero** modules and still prints that line. The driver uses
   `NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG` and raises if its `forward_loop` never ran.

## Gates on the driver's own output

- every tensor present exactly once in the rebuilt index
- each tensor's payload sha256-identical to what the exporter produced (`reshard.py` does this)
- one component per shard, never two (`reshard.py --plan`: all 296,475 tensors classified, 0 left over)
- the finished checkpoint serves

## State

`/opt/llm/runners/lh/lhbuild.py`. The PBS streaming path is **exercised and working** — index fetch,
shard-header parse, byte-range resolve, verified on layers 0/5/24/47. Not yet written: stage 1
capture, and `_export` (the NVFP4 packer) which is still a `NotImplementedError`.

Row budget for stage 1: the corpus is 1,814,667 tokens (`calibration-corpus.md`) but the capture must
fit on disk at ~1 GB per layer per 200k tokens. 200k tokens gives 200,000 × top_k 10 / 512 = **3,906
rows per (layer, expert) on average** and 49 GB of capture across 48 layers, against 291 GB free.
The tail is what `MAX_NOROWS` guards.
