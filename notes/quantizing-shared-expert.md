# Quantizing `shared_expert`: the safe lever, and what it actually buys

`mlp.shared_expert.{gate,up,down}_proj` is 0.24 B parameters read on **every** token — the shared
expert runs unconditionally, unlike the 10-of-512 routed ones. At BF16 that is 0.47 GB/token of a
5.61 GB budget.

Four other projects quantize it (`tcclaviger` MXFP4, `lvkaokao` MXFP8 W8A8-dynamic, `textclf`
TQ-4bit, `local-inference-lab/4p89` MXFP8 g32). **None publishes what it buys.** The technique is
not new; the measurement is.

## Why this one was low-risk, when the previous two were not

The scheme picked itself. `shared_expert` is `(640, 2560)` and `(2560, 640)` — **both dimensions
divisible by 128** — so it takes `FP8_PB_WO`, the scheme already running on 157 tensors of this
checkpoint. No new kernel, no new scale format, no E8M0 encoding.

Every one of those was a failure mode in the hyper-connection attempt, which needed MXFP8 because
`320 % 128 = 64`, and then found no sm_121 kernel that would accept the shapes. Choosing a layer
*because* it fits the proven path is the cheapest form of risk control available here.

Swept up in the same pass, same scheme, same eligibility: `ple.key_proj`, `ple.value_proj`,
`self_attn.indexer.index_qk_proj` — another 0.10 GB/token.

## What we deliberately left alone

`mlp.gate` is eligible — `(512, 2560)`, 128-divisible, worth 0.13 GB/token — and we did not touch
it. It is the **MoE router**. A perturbation there changes *which experts run*, rather than nudging
an activation by 2%; the failure mode is discrete and not obviously visible in perplexity. The
checkpoint author excluded it too. The offline gate confirms it stayed excluded while everything
else resolved:

```
mlp.shared_expert.gate_proj   excl=False  algo='FP8_PB_WO'
ple.key_proj                  excl=False  algo='FP8_PB_WO'
mlp.gate                      excl=True   algo=None        <- router, left BF16
```

Cheap levers you decline are worth writing down; otherwise the next person reads 0.13 GB/token
sitting unclaimed and assumes it was an oversight.

## Result

Round-trip quantization error **2.2490%** across 158 tensors — the FP8 E4M3 floor, matching every
other FP8 group in this checkpoint.


## Result: measured, and not worth shipping

| | `shared_expert` BF16 | `shared_expert` FP8 | change |
|---|---:|---:|---|
| c=1 | 36.3 | **37.0** | +1.9% |
| c=2 | 52.3 | 57.8 | +10.5% |
| c=4 | 73.0 | 76.6 | +4.9% |
| c=8 | **115.8** | 112.6 | **-2.8%** |
| NLL/token | **0.9628** | 0.9826 | **+2.06% worse** |
| tasks | 10/10 | 10/10 | — |

**We are not shipping it.** ~2% faster at c=1, *slower* at c=8, for a 2% quality regression — and
unlike the two levers before it, this regression is directional rather than noise:

| lever | NLL change | character |
|---|---|---|
| dense projections → FP8 | −1.8% | 5 of 6 chunks better — noise |
| `lm_head` → FP8 | −0.6% | 9 of 14 better, 5 worse — noise |
| **`shared_expert` → FP8** | **+2.06%** | **consistent, directional** |

That is very likely why the checkpoint author excluded `*.mlp.shared_expert.*` in the first place.
Four other projects quantize it; none published what it costs. Now there is a number, and the
number argues against it.

## The bandwidth model over-predicted, and that is the more useful finding

We projected **~40 tok/s** from 0.47 GB/token off a 5.61 GB budget — 8.4% of the bytes should have
bought ~8%. We measured **+1.9%**.

The roofline assumes every byte removed is a byte the GPU was waiting on. That holds for the dense
projections at `(10240, 2560)`, which are large enough to saturate bandwidth. `shared_expert` is
`(640, 2560)` — a **small matmul**, latency- and launch-bound rather than bandwidth-bound, so
removing its bytes frees time the GPU was not actually spending.

**So the roofline is an upper bound whose tightness depends on matrix size**, and we had been
treating it as a predictor. Every projection on this page should be read with that caveat; the
large-matrix levers (dense projections, `lm_head`) tracked it well precisely because they are
large.

## What we would keep

Nothing here. `fp8head` — dense projections + `lm_head`, without `shared_expert` — remains the
configuration we serve.
