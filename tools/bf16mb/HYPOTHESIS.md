# BF16 small-M GEMM microbench (written 2026-09-25 ~01:20, before the run)

Context (40-prof, step 3): the target's BF16 linears (hyper-connection mixers down [324x10240] / up [10240x320], shared
expert gate_up [1280x2560] / down [2560x640], router [512x2560], in_proj_ba [96x2560]) run in prod on cuBLAS
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x{1,2}` (32-thread blocks). In-model per call: 40.5 / 33.9 /
46.0 / 23.2 / 25.6 / 17.2 us vs byte floors 31.3 / 29.8 / 29.8 / 14.9 / 11.9 / 2.2 (220 GB/s). ~13 ms/step on main.

Standalone, M=4, weights rotated over >= 96 MiB so L2 never holds them, a CUDA graph of back-to-back calls:
- H1: torch F.linear (default cuBLAS) reproduces the in-model per-call times within +-15 % -> the kernel itself is
  the cost, not in-model contention. Outside: at the floor standalone -> the in-model excess is contention/ordering.
- H2: at least one alternative (cuBLASLt via preferred_blas_library, FlashInfer mm_bf16 / tinygemm_bf16, a Triton
  split-K GEMV) reaches <= 1.10x floor on the mixer down and router shapes. Expected saving if swapped in-model:
  2-4 ms/step (4-7 %). Outside: nothing beats cuBLAS by > 5 % -> BF16 lever is only the 8-bit weights (-8 %).

## Server A/B (45-bf16sk, written 2026-09-25 01:12, before the run)

Deterministic two-pass kernel (`fn_bf16sk.py`), standalone <= 1.06x floor on all six shapes at M=1/4/16. Replacing
cuBLAS in-model removes ~1.0 ms (mixer down) + ~0.6 ms (mixer up) + ~0.5 ms (in_proj_ba) + the ~190 splitKreduce
launches per step; shared/router gains are uncertain (contention-bound in-model).
- Expected c=1 decode: +2.5 % to +5 % tok/s (sk vs base), every start of sk above every start of base.
- c=4: +1 % to +4 %.
- Acceptance within +-2 pp (numerics change, drafter shares the kernel); within-arm output hashes identical across
  starts (the kernel is deterministic); cross-arm hashes are EXPECTED to differ (different reduction order).
- Outside: < +1 % -> the in-model win is eaten by contention/launch cost; > +6 % -> something else changed, check
  the path lines.

## Profile with FN_BF16SK=1 (47-profsk, written 2026-09-25 02:3x, after sk0/sk1 showed a per-cycle null at c=1)

A/B so far: c=1 per verify cycle base 59.1 ms vs sk 59.4-59.9 ms (null or slightly worse), acceptance 51.2 -> 48.7 %,
c=4 +3..5 %. The kernel swap should have removed ~2 ms/cycle. Candidates:
(a) the Triton kernels run slower in-model than standalone (mixer down expected ~31 us; if >= 38 us, in-model
    conditions such as L2 state, launch cost, the partials buffer or graph capture eat the gain);
(b) the kernels are fast, but the time moved elsewhere (e.g. the MoE aux-stream overlap changed, or the drafter's M=1
    path got slower);
(c) the lower acceptance means more drafter + verify work per accepted token (already folded into ms/cycle, so it
    cannot explain a per-cycle null).
Expected: per-cycle BF16 category <= 12 ms (was 14.6 + splitKreduce) if (b); >= 14 ms if (a).

## Clock check (written 2026-09-25 ~03:3x, after 47-profsk)
47-profsk: Triton mixer down 41.1 us in-model (never overlapping a side stream) vs 30.3 standalone and ~34 in the idle
in-worker bench (2c). H: under sustained c=1 decode the SM and/or memory clock or the power cap drops >= 15 % below
the short-burst level -> the whole-step slowdown vs benches is clock/power, not the kernels. If the clocks during
decode equal those during a microbench burst, the cause is elsewhere (CPU DRAM traffic on the unified memory,
TLB/page layout of the model weights).

## Producer->consumer micro-chain (pairbench, written 2026-09-25 ~03:35, before the run)
Consumer: mixer down [336x10240] (Triton det 2-pass and cuBLAS), M=4, weights rotated. Producers before each call:
P0 none, P1 tiny elementwise ([4,10240] add, kernel boundary only), P2 fp32 write of 3 MiB (GDN-state-sized per layer),
P3 fp32 write of 12 MiB, P4 read-only 12 MiB reduction. Marginal consumer cost = graph(P+C) - graph(P alone).
- (a) dirty-L2 write-back: P2/P3 raise the consumer's marginal cost by >= 5 us (P3 > P2), P4 does not.
- (b) kernel-boundary ramp: P1 raises it by >= 5 us over P0.
- Neither -> the in-model overhead needs the real sequence (in-worker replay).

## Bimodality check (written ~03:40 after pairbench showed P0 42.1 vs P1 30.8 us for the same consumer)
H: the same mixer-down chain repeated 40x is bimodal, ~30 and ~41 us (ratio ~1.33 = LPDDR5x 8533/6400 MT/s bins),
independent of code -> the "in-model excess" is a DRAM frequency/power state. Unimodal -> pairbench noise is
something else.

## "Slow after out_proj" (pairbench2, written 2026-09-25 ~07:50, before the run)
Trace fact (both arms, r=0.88): mixer down takes 34.6 us at the MLP-side site but 46-53 us right after the FP8
blockwise out_proj/o_proj (grid [1,20,1]), after GDN AND after QSA layers alike. Standalone, M=4, weights rotated:
C0 mixer down alone; C1 out_proj(FP8 blockwise, cutlass_scaled_mm, the in-model call) -> mixer down;
C2 out_proj -> tiny kernel -> mixer down (the in-model order has _hc_combine_norm between); C3 a BF16 read of the
same 15.7 MB (Triton) -> mixer down.
- H-a: the CUTLASS FP8 blockwise kernel itself leaves a state that slows the next read: C1/C2 marginal >= 40 us,
  C0 and C3 ~31 us.
- H-b: any preceding large read does it: C1, C2 and C3 all >= 40 us.
- Neither (all ~31 us): the effect needs the real sequence (e.g. GDN state or the MoE aux-stream join) -> in-worker.

## Profiled-faster-than-unprofiled check (written 2026-09-25 ~08:30, before running, on warm prod)
40-prof and 47-profsk both measured unprofiled decode(400) at 59.6/59.8 ms/step and the profiled decode(150) at
55.0/55.6, minutes apart in one server. On the warm prod server (hours up), same prompt, same probe code:
decode(150) x3 and decode(400) x3, alternating.
- If decode(150) ~= 55 and decode(400) ~= 59.5: the gap is request length/content, not the profiler; no anomaly.
- If both ~= 59.5: the profiler speeds decode up by ~4.5 ms/step -> a host-side scheduling interaction (the GPU idles
  ~7 ms/step unprofiled vs 2.4 profiled); next = measure unprofiled GPU idle.
- If both ~= 55: warm-up drift explains the probes' 59.6 (the probes ran on a fresh server).

## Decisive re-test at reduced KV (written 2026-09-25 ~09:25, before the run)
Finding 238 (per-cycle null) ran with 33.5 GiB KV while the checkpoint-mapped PLE paged (~30 major faults/step).
Same A/B at --kv-cache-memory-bytes 4 GiB, PLE warmed after ready (method taken from the running plekv4 baseline),
1 start per arm. c=1 ms per verify cycle (ms/tok x accept length):
- If sk beats base by >= 2 % per cycle -> paging masked kernel gains -> re-run 237, 235, the profile.
- If |diff| < 1 % -> finding 238 stands; paging only shifted absolute numbers -> correction notes, no re-runs.

## TLB-aftershock test (tlbbench, written 2026-09-25 ~16:55, before the run)
Warm trace: the slow spots follow the eager GDN/attention section, not graph position (mixer down elsewhere at
graph position 1-2 is 34 us; after out_proj at position 3/5 it is 48-50 us; out_proj itself 99 us vs 70 standalone).
Standalone, box idle (prod down): producer = a gather touching one 4-byte element every STRIDE bytes across SPAN GiB
of a large device buffer (TLB sweep), then the in-model FP8 out_proj (cutlass_scaled_mm, [2560x6144]) and mixer down.
- H-TLB: after a sweep of >= 2,000 distinct 2 MiB regions, out_proj >= 88 us (vs ~70) and the following mixer down
  >= 42 us (vs ~31); no effect after a sweep of a small span (<= 64 MiB).
- Refuted if the consumers stay within +10 % of the no-producer numbers at every span.

## TLB refuted; memory power-state test (dvfsbench, written ~17:10, before the run)
tlbbench: 2 MiB-stride sweeps of 6,144 regions / 12 GiB -> no effect (83.8 / 34.7 vs 83.4 / 34.6 baseline); heavy
64 KiB sweeps made out_proj FASTER (73.6-75.9). Eager-with-idle baseline out_proj 83 vs 70 in a busy graph.
H-DVFS: after ~300 us of DRAM-LIGHT GPU work (spin kernel torch.cuda._sleep; or 30 tiny L2-resident kernels) the next
big weight read is slow (out_proj >= 85 us, mixer after >= 40 us); after ~300 us of DRAM-HEAVY work (read a 128 MiB
buffer) it is fast (out_proj <= 75, mixer <= 36). Refuted if the three producers give the same consumer times (+-5 %).

## nsys node-level trace of a warm step (written ~18:15, before the run)
Lead: a BF16 GEMM that normally takes 23 us takes 177 us as the first kernel of a graph launch; slow spots sit at
the start of graph launches after eager sections. nsys --cuda-graph-trace=node shows per-node timing plus the CUDA
API side (cudaGraphLaunch duration, graph upload, memory ops) that the torch profiler hides.
- H: the first node(s) of the post-eager graph launches start late or run long while graph-launch work
  (upload / memory-pool map) completes; the host-side cudaGraphLaunch of those pieces takes >= 50 us.
- Refuted if node timings under nsys show the same in-kernel slowness with no launch-side activity nearby -> the
  cause is inside the kernel's execution (memory system state), for which a hardware counter profile (ncu) of that
  one kernel in place is next.

## nsys result + dirty-L2 test v2 (l2dirty, written ~18:35, before the run)
nsys (node trace, warm): GDN out_proj 102.0 us, QSA o_proj 75.7 us, back-to-back (0.5 us gap); graphs launched ~75 ms
ahead -> graph-launch work refuted. Directly before GDN out_proj: fused_sigmoid_gating_delta_rule (42 us, grid
[1,4,48]) which reads a 3 MiB fp32 state and writes up to 4 speculative states (<= 12 MiB) in 42 us -> faster than
DRAM, so the writes sit dirty in the 24 MiB L2 and are written back during the next kernels.
Test: CUDA graph of [read an L2-sized buffer (makes it resident), overwrite X MiB of it (dirty, fast), out_proj,
mixer down] x N, kernel durations from the torch profiler (CUPTI) -> no launch-latency artifact.
- H-dirty: out_proj grows ~+10-15 us per 3 MiB dirtied (X = 0/3/6/12 MiB) and the mixer after it grows too; at
  12 MiB out_proj >= 90 us (clean ~70).
- Refuted if out_proj stays within +5 us across X.
