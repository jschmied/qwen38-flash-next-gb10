# vllm#56824 on our GB10 — hypothesis, written 2026-09-24 before any run

**Reported:** Flash-Next NVFP4 on GB10 (v0.29.0 plus the reporter's PLE disk-offload overlay). The flags are fp8 KV,
util 0.70, 256 seqs, 32k, MTP 3, compile mode 0 + FULL_DECODE_ONLY [4..24], no prefix cache, no FlashInfer
autotune and the V2 runner. During KV sizing and graph capture, host memory collapses (29 GiB in 9 s) and the
driver logs `NV_ERR_NO_MEMORY`, while `MemAvailable` still shows 22 GiB. The box kernel-OOMs without a guard.

**Our stack:** main 1ea7c63f4, our checkpoint-mapped PLE (the table is not resident; comparable to their disk
offload), the mtpfp4 checkpoint. Prod (util 0.90, 16 seqs, BF16 KV, PIECEWISE, which captures nothing) starts fine.

**Arms, sequential, guarded** (SIGKILL when MemAvailable < 5 GiB for 2 × 0.5 s samples):
- R0: the control, their util/seqs/len, PIECEWISE, BF16 KV;
- R1: their flags;
- R2: R1 in PIECEWISE (graph capture isolated);
- R3: R1 with 16 seqs (sequence count isolated).

**Expected:**

| cell | expected |
|---|---|
| R0 | ready and serves; min MemAvailable ≥ 20 GiB |
| R1 | residual peak ≥ 10 GiB above R0's. Guard fires or startup fails: plausible, ~50 %. If it starts, it serves |
| R2 vs R1 | if graph capture is the source, R2 stays near R0 |
| R3 vs R1 | if the startup footprint scales with max-num-seqs (their workaround direction), R3 is smaller |
| NVRM `NV_ERR_NO_MEMORY` | may appear even on healthy starts (thread consensus: low precision) |
| FULL_DECODE_ONLY on this model | may fail to capture outright (PLE/QSA in graph). That would be a different failure; record it, but it is not #56824 |

An outcome outside these ranges means I check the instrument first (sampler cadence, phase detection).
