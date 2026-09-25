# Cold window = PLE first-touch faults? (written 2026-09-25 ~08:40, before the run)

Known: a fresh server decodes ~59.6 ms/step for its first few hundred tokens, warm prod 53.7 (speed-of-light 4c).
DS4.1 saw the same shape: first decode tracking the Engram (= our PLE) read time. PLE is read in place from the mapped
checkpoint (`ple_pageable.py`, MADV_RANDOM, one fault per 4 KiB page, CPU prefetch is only a hint).

Arms (fresh start each, prod config, 2 starts each, alternating): `base` (stock) vs `pop` (FN_PLE_POPULATE=1:
MADV_WILLNEED + MADV_POPULATE_READ of all 47.7 GiB at load). Probe: 6 different 300-token requests.
- H1 (first-touch faults): base request 0 is >= 3 ms/step slower than requests 3-5, with worker minor+major faults
  per step in request 0 >= 10x those of requests 3-5; pop has request 0 within 1.5 ms/step of requests 3-5, faults per
  step ~0 from the start, and requests 3-5 equal in both arms (+-1 %).
- H0 (something else): base shows the cold window without a fault-rate difference, or pop keeps the cold window
  -> allocator/compile/cudagraph/power warm-up; next = profile the cold window.
- Also recorded: PLE page-cache residency when the server becomes ready (did loading evict it?).

## Round 2: drop the loaded shards from the page cache (written 2026-09-25 ~08:55, before the run)
Round 1 base0: at ready the PLE was 0.03 of 47.68 GiB resident (loading the other weights evicted it), then ~30 major
faults + 0.3 MB NVMe per step, 60 -> 58.7 ms/step over 1,800 tokens. Arms: `drop` (FN_LOAD_DROPCACHE=1: fadvise
DONTNEED on each non-PLE shard after its tensors are consumed) and `droppop` (drop + FN_PLE_POPULATE=1). Each start
follows a start that left the PLE cached (as a prod restart does).
- H: `drop` finds >= 40 GiB of PLE resident at ready, ~0 major faults per step, request 0 within 1.5 ms/step of warm
  prod (53.7); `droppop` equal to `drop` (+-1 %) with 0 faults. Load time unchanged within +-10 %.
- Outside: `drop` still loses the PLE -> something else evicts it (KV/graph allocations of the unified pool, not the
  page cache of the loaded shards).

## Round 1 result + round 2 redesign (09:00)
Round 1 (base0, pop0): PLE 0.02-0.03 GiB resident at ready in BOTH arms; FNPOPULATE populated all 47.7 GiB at 08:28
(rc 0), but KV-cache allocation/profiling afterwards (gpu_memory_utilization 0.9 -> 33.5 GiB KV on top of 72.4 GiB
weights) evicted it again. ~30 major faults/step, 60 ms/step in both. Weights + PLE = 120.1 GiB of 121.6.
Round 2 `warmext` (1 start): stock server; at ready the probe records meminfo + residency, reads the 47.7 GiB PLE
sequentially into the page cache, records again, then runs the 6 requests.
- H: if the page cache can hold >= 40 GiB of PLE after ready (KV committed lazily), request 0 runs at 53.7 +- 1.5
  ms/step with < 3 major faults/step -> the cold window is PLE page-cache absence, fix = warm after ready.
- If the warm read cannot keep the PLE resident (MemAvailable too small once KV is committed), residency stays low
  and faults continue -> the table cannot be fully resident beside a 33 GiB KV budget; then size the KV cache.

## Round 3: KV reduced to 4 GiB (user: "going into paging destroys every measurement") — written 09:15
All earlier runs (incl. findings 234-238) used gpu_memory_utilization 0.9 = 33.5 GiB KV: weights 72.4 + KV 33.5 leave
~10 GiB for page cache vs the 47.7 GiB PLE -> ~30 major faults/step. Now `--kv-cache-memory-bytes 4 GiB` (~76k
tokens; enough for c<=4 probes), ~40 GiB free for page cache. One fresh start per arm.
- `kv4` (no warm): PLE low at ready (the load evicts it), MemAvailable >= 35 GiB; faults/step fall across the 6
  requests as rows are cached, ms/step reaches <= 55 by request 5.
- `kv4warm` (external sequential warm after ready): >= 35 GiB of PLE resident after the warm, request 0 at
  53.7 +- 1.5 ms/step with < 3 major faults/step.
- Outside: kv4warm still faults >= 10/step with the PLE resident -> the faults are not page-cache misses (e.g.
  ATS/page-table population per process) -> MADV_POPULATE_READ after the warm is the next arm.

## Round 3 result + round 4 (written 09:45, before the run)
kv4: PLE 0.03 GiB at ready (MemAvailable 33.3, Cached 31.0 = the loaded weight shards), ~30 majflt/step, 60 ms/step.
kv4warm: warm read 47.7 GiB in 22 s but only 16.7 GiB stayed resident (Cached 29.6), 15-26 majflt/step, 57.5-59.5.
Round 4 `dropwarm`: FN_LOAD_DROPCACHE=1 + KV 2 GiB + warm; full /proc/meminfo at ready / after warm / end.
- H: the weight-shard cache is gone at ready (Cached < 5 GiB), warm keeps >= 28 GiB of PLE resident, faults/step
  roughly scale with the non-resident fraction (<= 12).
- If residency >= 45 GiB (>= 95 %): paging-free measurement setup found -> run the decisive FNBF16SK re-test with it.
- If residency stays <= 35 GiB: a fresh server cannot hold the table beside the weights; find what warm prod did
  (full meminfo of warm prod) before any re-run.

## Round 5: who takes the PLE faults — CPU prefetch or GPU? (written ~10:25, before the run)
Fault latency measured on the idle box: one 4 KiB random read ~60 us by any route (O_DIRECT 64, buffered 58, mmap
fault 58; cached minor fault 1.6 us); in parallel the SSD gives ~50k IOPS, the prefetcher's numpy pattern 231k rows/s
at 64 threads. Yet the server pays ~0.2 ms per major fault per step (~30/step -> ~6 ms), 3x a serial SSD read.
FN_PFTIME=1 (1 fresh start, KV 4 GiB, no warm, coldprobe): per step, host prepare->inputs->touch-done times, GPU time
of the gather kernel (CUDA events), whether the GPU started the gather before the touch finished.
- H-gpu (GPU loses the race and faults serially): gpu_started_before_touch_done >= 0.5; gather_gpu_us p50 >= 2000 us;
  gather time ~ faults x (60-200 us).
- H-cpu-slow (prefetch finishes first but takes long): gpu_started... < 0.2, touch_ms p50 >= 3 ms, gather short.
- Neither (gather short AND prefetch early): the 6 ms is not in the lookup at all -> look elsewhere (host stalls).

## Round 6: the gather waits for the CPU touch (FN_PLE_SYNCTOUCH) — written ~11:05, before the run
Option 2 (user: the PLE must not take KV memory). The lookup waits (<= 50 ms, GIL released) until the prefetch thread
has touched the step's rows with 64 threads, so the GPU finds the pages mapped instead of faulting them serially.
Arms (1 fresh cold start each, KV 4 GiB, no warm, FN_PFTIME=1 on both): `base` vs `sync`.
- H: sync gather_gpu_us p50 <= 300 us (base ~4,500); lookup_wait_ms p50 <= 1.5 ms; ms/step request 0 <= 56.5
  (base ~60, warm floor 53.7); per-request output hashes identical between arms (timing-only change).
- Outside: wait_ms ~ 4-5 ms (CPU touch no faster than GPU faults) -> the parallel-fault speedup does not materialise
  in-process (page-cache lock / mmap_lock contention) -> host pread into a pinned buffer instead of touches.

## Round 6 result + round 7 (written ~11:45, before the run)
Round 6: outputs identical (6/6 hashes); sync gather 107-121 us (base ~4,000) but ms/step WORSE (61.2-64.4 vs
58.8-61.0): the stock touch() handles < 4096 rows serially in one thread -> 5.6 ms of CPU faults while the GPU idles
(lookup_wait ~45 ms incl. the host's one-step lead); true faults/step ~56 (base counted only the CPU's ~25).
Microbench (56 cold rows): serial 4.8 ms; numpy split 16 tasks 1.66 ms; per-page MADV_POPULATE_READ 32 tasks
1.37 ms = ~40k IOPS, near the SSD's ~50k parallel ceiling.
Round 7: sync touch = per-page POPULATE_READ on 32 pool tasks. Same arms (1 cold start each, KV 4 GiB).
- H: sync touch_ms p50 <= 2 ms; gather <= 300 us; ms/step <= base - 1.5 ms (base ~59.5 -> sync <= 58); hashes equal.
- Outside: sync >= base -> the lost host lead (the lookup now waits for the previous step to finish) costs more
  than the faults save -> move the wait onto the GPU (stream wait on a host-set flag) so the host keeps its lead.

## Round 7 result (12:30)
sync (per-page POPULATE_READ, 32 tasks): outputs identical 6/6; gather 100-112 us; touch 3.7 ms for ~57 pages
(in-server ~15k IOPS vs 40k on the idle box) -> ms/step 58.6-59.5 vs base 58.0-60.9: null. Outside H (touch <= 2 ms).
Suspect: direct reclaim in the fault path (MemFree ~3 GiB in-server vs 117 GiB in the microbench).

## Round 8: is the in-server fault path doing direct reclaim? (written ~12:40, before the run)
Round 7 sync touch: 57 pages in 3.7 ms in-server (~15k IOPS) vs 1.37 ms on the idle box (~40k). One cold start, KV
4 GiB, FN_PLE_SYNCTOUCH=1 + FN_PFTIME=1; per request, /proc/vmstat deltas per step: pgscan_direct, pgsteal_direct,
allocstall, pgscan_kswapd, workingset_refault_file, compact_stall.
- H (direct reclaim): pgscan_direct >= 50 pages/step and allocstall > 0 per step (i.e. the faulting threads reclaim
  inline), or kswapd scanning at >= the fault rate with refaults (thrashing).
- Refuted if pgscan_direct ~ 0 and allocstall ~ 0: the slowdown is elsewhere (CPU contention of the 64 touch threads
  with the server's own threads, or the SSD queue shared with other I/O).

## Round 8 result + round 9 (written ~13:00, before the run)
Round 8: pgscan_direct/pgsteal_direct/allocstall/pgscan_kswapd all 0 per step, MemFree 6.8 GiB -> NO reclaim; the
in-server slowness is not memory pressure. Microbench: the Python pool doing per-page madvise = 1.8 ms; a C helper
(libfnpopulate.so, one ctypes call, 16 pthreads, GIL-free) = 0.60 ms for 57 pages (~95k pages/s).
Round 9: sync touch through the C helper. Arms base vs sync, 1 cold start each, KV 4 GiB.
- H: sync touch_ms p50 <= 1.2 ms; gather <= 300 us; ms/step sync <= base - 2 ms; hashes identical.
- Outside: touch stays ~3.7 ms in-server with the GIL out of the path -> contention is in the kernel/SSD queue
  under the server's load, not in Python.

## Round 10: confirmation, 2 starts per arm + warm second pass (written ~13:55, before the run)
Round 9 (1 start/arm): sync+C -2.2 ms/step cold, 6/6 requests, hashes identical. Now base/sync alternating, 2 starts
each; probe = 6 cold requests then the same 6 prompts again (rows cached + mapped by the first pass).
- H-cold: sync < base on every paired request in both start pairs, mean delta -1.5 to -3 ms/step.
- H-warm: second pass majflt ~0 in both arms; |sync - base| <= 0.5 ms/step (the wait costs little once nothing
  faults). Outside: sync warm >= base + 0.5 -> the lost host lead costs real time in steady state -> gate the wait
  (only when the previous steps faulted) before any promotion.

## Round 11: auto mode (written ~15:05, before the run)
Round 10: unconditional wait cold -2.05 ms/step (12/12 paired requests), warm +2.3 ms. Auto: wait only while the
EMA of measured major faults per step >= 12 (starts cold = waiting). Arms base vs auto, 1 start each, cold + warm pass.
- H: auto cold within 0.7 ms of round-10 sync (57.5 -> <= 58.2); auto warm within 0.5 ms of base warm (~54.9);
  hashes identical; the FNPFTIME summaries show fault_ema >= 12 in the cold pass and < 12 in the warm pass.
- Outside: auto warm still >= base + 1 ms -> the gate does not switch off (EMA stuck) or waits still happen; auto cold
  ~ base cold -> the gate switches off while cold (undercount).
