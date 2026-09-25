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
