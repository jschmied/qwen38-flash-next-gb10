# Research agents on the measurement box invalidated five arms

2026-08-31. Between 20:18 and 21:36 the fixed-work agent loop appeared to become erratic: the same
configuration returned 31.93, then 38.96, then 57.61 ms/tok. Elaborate explanations followed — ring
capacity, draft depth, bimodal startup state — and all of them were fitted to contaminated points.

**The cause was two research subagents running on the same machine.**

| time | arm | ms/tok | subagent active |
| --- | --- | ---: | --- |
| ≤19:44 | 11 arms, every config | **all reproducible to ~1%** | no |
| 20:03 | MTP4_a | **31.93** | no |
| 20:18 | MTP4_b | 38.96 | yes (20:15-20:33) |
| 20:33 | MTP6_a | 52.11 | yes |
| 20:49 | MTP6_b | **71.85** | just after |
| 21:05 | MTP3_a | **32.48** | no (between agents) |
| 21:20 | MTP3_b | 49.08 | yes (21:19-21:35) |
| 21:36 | MTP4_c | 57.61 | yes |
| 21:52 | **MTP4_d** | **31.47** | **no** |

Every quiet arm is fast; every arm inside or just after an agent window is inflated, by 24% to 83%.
`MTP4_d` was run as a stated prediction — *"if contamination explains it, this lands near 32"* — and
returned 31.47 against MTP4_a's 31.93.

## Why "read-only" was not enough

Both agents were explicitly told **not to touch the GPU**, and neither did. They ran ~120 tool calls
each: grepping the venv, reading source, reading safetensors headers across a 123 GB checkpoint.

On GB10 that is not harmless. **CPU and GPU share one memory pool**, and this model's PLE gather
depends on **47.68 GiB of host page cache**. Bulk file reads evict exactly the pages the offload
worker needs. The GPU was never touched; the thing the GPU was waiting on was.

**On unified memory, read-only is not load-free.** Isolation has to be stated in terms of the
*resource* — host memory, page cache, I/O, CPU — not the device.

## Rule

While arms are running: no subagents, no manual probes, no bulk reads under `/opt/llm`, no
downloads. This is the third contamination in one session — a manual NIAH probe against a live arm,
a second vLLM server from a fall-through chain, and now research agents — and all three were
self-inflicted while the benchmark harness itself behaved correctly.

Detection, cheaply: keep a control config in the rotation and interleave it. Eleven arms before
20:00 replicated to ~1%, which is the only reason the break was visible at all. A run without an
interleaved control cannot distinguish "this config is slow" from "the box was busy".

Related: [[evidence-standard]], [[failure-modes]].
