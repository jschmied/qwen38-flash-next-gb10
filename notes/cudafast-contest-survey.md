# What `cuda.fast` is, and the six things in it we do not already have

2026-09-13. The user pointed at `yukon.org/mlxfast?platform=cuda`. It is worth a note because the
target is **our model on our box** — `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL on a pinned DGX
Spark, sm_121, nvcc 13.0.88 — and 88 promoted submissions are ~943k characters of measured GB10
engineering. The leaderboard itself is of no interest; the notes attached to it are.

## What it actually is

An **AI-agent optimization contest** run by Eigen Labs on their Yukon platform, not a library, a
kernel set or a runtime. `?platform=cuda` swaps between two sibling tracks (`mlx.fast`, Apple, closed;
`cuda.fast`, DGX Spark, active). **Nothing MLX is involved** — the name is lineage from a shared
ancestor repo, the two 125B tracks were created four minutes apart, and there is no MLX code in the
CUDA track. The "Model" column is which LLM drove the solver.

The engine under optimization is **`ds4`/DwarfStar**, antirez's personal engine, MIT, whose own README
says *"deliberately narrow… consider it beta quality"*. Qwen3.8-Flash-Next support is not upstream at
all; it lives in a private fork vendored into the challenge repo.

## The headline, deflated

`composite = prefill_gain^0.25 × decode_gain^0.75`, and the baseline is **the same ds4 engine with
speculation off**, same box, minutes earlier. Never vLLM, never llama.cpp. Recovered from their own
control tables: baseline decode **15.5 tok/s**, frontier **33.4**.

Turning MTP on — the thing the contest is named for — bought **+7.0 % composite**. Everything from
1.070 to 2.425 is hand-optimizing an untuned engine. N is **one prompt** (`botany`, 1024 prefill /
128 decode); the fidelity gate permits 10 % token divergence though the culture is stricter; decode
is saturated at the top (32.3–33.7 TPS across the top ten) and recent movement is all on the
0.25-weighted prefill axis.

**The one ratio that governs every transfer verdict:** their frontier moves 2.18 GB/token at 33.4
tok/s = **73 GB/s, about 30 % of the GB10 roof**. Ours moves 9.7 GB/token at 17.1 tok/s = **166 GB/s,
about 70 %**. Their engine is latency-bound; ours is not. Most of their campaign is climbing to where
we already are.

## Corroboration worth keeping

* Their independent streaming roof: **235.8–237.6 GB/s, 86 % of the theoretical 273** — an outside
  confirmation of our own 220–240 GB/s figure.
* Their ranked noise floor was **retracted mid-corpus** from ~0.3 % to 0.8–1.0 % within a box and
  2.3 % across boxes. The reason is the useful part: they had measured the floor off the *control*
  leg, but control and candidate are differently-shaped computations that do not respond to thermals
  proportionally, so the ratio does not cancel. One tree that was absolutely faster on both legs was
  rejected 0.62 % below the leader purely because its control leg drew faster.
* **"Kernel busy" ≠ GPU wall**: naive sum 41.0 ms vs union-of-intervals 39.9 ms in a 41.7 ms round —
  1.1 ms of overlap that turned a real 1.8 ms idle into an apparent 0.7 ms and "hid the main result
  for a day". Independent confirmation of [[decode-c1-idle-piecewise]].
* **Graph-replayed kernels carry no kernel name in CUPTI**, so summing a kernel table omits most of
  decode and reports ~98 % occupancy regardless. They had to merge `KERNEL` with `GRAPH_TRACE`.

## The six that transfer

1. **The MTP draft head has no causal state over the prompt.** On ds4 the head's attention K/V and the
   QSA indexer tape were **all zeros across the entire prompt** — it drafted blind, and correct target
   verification hid it completely. Fixing it (run the head's input mixer per prefill chunk) was their
   single largest jump, **+2.58 %**, and acceptance moved most on long prose editing (55/71 → 62/64).
   Our own MTP acceptance on agent traffic is only **41–50 %**. Cheapest check needs no benchmark:
   read whether vLLM's MTP proposer runs the draft layer over prefill tokens or merely allocates its
   KV slot. If it only allocates, bucket per-position acceptance from an existing agent run and look
   for depressed acceptance in the first turn after a cold prefill.
2. **GDN butterfly re-tiling, and the hardware fact under it.** **Shuffle retires at a quarter of the
   FP32 FMA rate on GB10**, so a 32-lane butterfly where an 8-lane one suffices dominates issue. One
   value row per eight-lane group instead of per warp: 40 shuffles per warp-token → 6, SASS `SHFL`
   40→6, issued instructions 217→152, kernel **54.10 → 40.39 ms (−25.3 %)**. GDN is 36 of 48 layers
   and this is a different lever from our fla-core kkt+solve fusion. Cheapest check: dump SASS for
   `chunk_gated_delta_rule` and count `SHFL`.
3. **QSA `float4` key rows.** Consecutive lanes hold different keys, so scalar per-channel reads
   fetched 32-byte sectors and used 4 bytes of each. **Prefill 359 → 174 ms, −52 %** — the largest
   measured kernel win in the corpus, held off the leaderboard only by a harness cap on
   single-submission gains. Directly on our declared TTFT goal.
4. **The MoE `down_partial` round-trip.** `moe_down_mma` writes a partial that `moe_down_combine`
   reads back: 104.9 MiB out + 104.9 MiB in per layer × 48 ≈ **10.1 GB per prefill forward, ~37 ms at
   273 GB/s**. Arithmetic only — nobody with a box has taken it, and it is blocked for them by
   bit-exactness (atomics reassociate) which is a constraint **we do not have**.
5. **PLE cold-fault amplification under speculation.** A **rejected draft token names PLE rows no
   committed token ever names**, so a serial control leaves them cold and the candidate eats the
   faults inside its timed window. Cold major fault 324 µs vs 5 µs warm. Two useful negatives:
   `posix_madvise(WILLNEED)` at gather time gave nothing, and a background-thread warm was *worse*.
6. **O_DIRECT with parallel readers for checkpoint load.** Their pathology (buffered 0.63 GB/s) does
   **not** reproduce here — measured on this box today: buffered 1 reader **1.7 GB/s**, O_DIRECT 1
   reader **2.8**, O_DIRECT 4 readers **4.9**. Still a real **2.9×**: a 126 GiB checkpoint goes ~76 s
   → ~26 s, which is pure restart cadence for A/B work.

## What does not transfer, and why that is the interesting part

Most of their 1.07 → 2.42 is **class E**: byte-wise → word-wise quant decode, a serial 512-expert
metadata scan replaced by a warp prefix scan (31.8 → 4.5 µs), a full bitonic sort of 512 router logits
to read the top 10, compile-time format specialization, a 993 KB D2H per round for a host-side argmax,
48 redundant activation-quantize launches per step, a dead post-RoPE K store. These are the defects a
hand-written engine accumulates and a mature runtime does not have.

Also not ours: graph-island splitting (vLLM already captures piecewise; their finding is that one
1497-node capture cost 0.65–1.0 ms of host time per replay with the GPU idle, unpipelined); PDL
(already in vLLM/CUTLASS, and routed-MoE edges cannot use it at all because the weight addresses *are*
the router output); the draft-vocabulary shortlist (ours is det-135, +6.4–6.8 %, ranked on real agent
traffic rather than BPE prefix order); and batch-1 MoE tile padding, block-size and bank-pitch recipes
— right principle, wrong site, since our MoE is 0.8 % of decode wall.

**Their depth-1 MTP verdict is inverted for us, and it is also stale on their own evidence.** On ds4
each extra verified row draws 20 of 512 expert slots per layer with ~2 % cross-row reuse, so row count
*multiplies* the dominant traffic term. Our 9.7 GB/token is dominated by shared dense BF16 weights, so
the extra row is nearly free — which is why our optimum is k=2–3 and theirs is 1. Their own numbers
disagree with each other anyway: an estimate of 23.17 ms for a marginal verified row (PR 355) against
a later *measurement* of +5.2 ms (PR 431), with nobody re-sweeping depth after the campaign. The
forward-looking implication for us: **if we move to a fully-quantized checkpoint where MoE dominates
the byte count, our optimal k may drop toward theirs.**

Three hardware facts worth carrying regardless: smaller blocks win on GB10 at these shapes (block-size
gradient ~1 % per halving, pointing smaller); fusion is not free at 48 SMs (pairing GDN qkv+gate with
QSA k+v scored 1.085 against 1.138; a shared-expert tile 8→16 rows dropped prefill 570→564 tok/s and
8→32 dropped it to 498); and the static shared-memory cap is 49,152 B/block with the driver **silently
declining** kernels over it, so a fallback can be credited with a timing unless you assert on it.
