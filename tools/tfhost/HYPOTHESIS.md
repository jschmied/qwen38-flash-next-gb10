# TensorFold host-side cost: draft-chain syncs and the eager concurrent path (2026-10-01)

Question (tensorfold-opportunities "Speed-of-light agenda"): how much of TF 0.6.0's decode time is host-side —
(1) the per-draft-step round trip in `draft()` on the serial path, (2) eager launches on the multi-stream path.

Arms, EXL3 3.05bpw, greedy, 512 tokens of code after a 64-token warm-up, one process per arm, timing runs without
nsys, then one nsys run per arm for GPU duty cycle (busy-union / capture window) and syncs per token:
- S1 serial, graphs on (default); S0 serial, graphs off; M2 streams=2 at c=2; M4 streams=4 at c=4.

Hypotheses (ranges before measuring):
- S1 GPU duty cycle 75–90 %; host gap per draft step 0.1–0.4 ms. If duty ≥ 90 % the draft-chain lever is < 5 % and
  not worth a restructure; if ≤ 80 %, worth a design note.
- S0 vs S1: graphs worth 10–30 % ms/token on the serial path (TF launches many small kernels per layer).
- M2 / M4 GPU duty 40–75 %; aggregate tok/s M2 1.3–1.7× S1, M4 1.8–2.8× S1. A duty ≤ 70 % at c=2 says graphs for the
  concurrent path are worth pricing.
- Outputs: S1 and S0 greedy token hashes identical (TF's exactness contract); M2/M4 streams equal to solo per
  contract (same prompts) — a mismatch voids nothing here but is a finding.
Void: an arm whose log lacks its `ARM` json line, or whose engine reports a different streams/graphs setting.
