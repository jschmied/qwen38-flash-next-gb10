# Build driver — design, written 2026-09-10 while cgab holds the GPU

**Constraint honoured while writing this: no bulk I/O.** `cgab` has four server starts left and each
needs ~48 GB of *pinned* host memory for the PLE offload worker. `tiecensus` already died with
`CUDA error: out of memory` while `free` reported 118 GiB available, because 105 GiB was page cache from
a hash pass. Reading the 335 GiB source now would reproduce that. Nothing below has been executed.

## The loop

For layer i in 0..47:
  1. **pull** layer i's tensors from PBS by name (they are scattered across shards — layer 0 spans
     shards 1..112 — so fetch by tensor, not by shard)
  2. **run** the cached calibration activations through layer i, in eager mode
  3. **calibrate** its experts on their own routed rows
  4. **export** packed NVFP4 + scales
  5. **capture** layer i's output as layer i+1's input, **drop** layer i's weights
  6. **checkpoint** progress so a kill resumes at the layer it reached

Peak memory is one layer plus the activation cache. Peak disk is one layer plus the growing output.

## The three things this session proved it must do

1. **Feed `down_proj` from the in-loop forward.** Its input is the expert's *intermediate* activation,
   not the layer input. Our captured tensor cannot calibrate it — the 2026-09-10 07:2x check used
   weight-only max purely to exercise the shape path. Calibrating `down_proj` from the wrong tensor
   would silently produce a half-max-calibrated model.
2. **Count and report LH / thin / no-rows per layer.** On one layer with 8,192 rows: 297 experts got
   ≥64 routed rows, 194 got 1–63, and **21 got none** and fell back to weight-only max. That fallback is
   invisible in the output checkpoint. The driver must emit the three counts per layer and refuse to
   finish silently if the no-rows bucket is large.
3. **Verify by iteration count, never by "Calibration complete."** `NVFP4_DEFAULT_CFG` with
   `algorithm=local_hessian` calibrates **zero** modules and still prints that line. Use
   `NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG` and assert `MSE weight calibration: N it` with N > 0.

## Gates the driver enforces on its own output

- every tensor present exactly once in the rebuilt index
- each tensor's payload **sha256-identical** to what the exporter produced (`reshard.py` does this)
- one component per shard, never two (`reshard.py --plan` classified all 296,475 tensors, 0 unclassified)
- the finished checkpoint **serves**

## Still unmeasured

The forward passes. Everything else is measured: experts-only calibration+export is ~35 min for 48
layers, peak GPU 0.1 GiB, and the source is verified 144/144. The forward cost is the one number the
build itself will produce.
