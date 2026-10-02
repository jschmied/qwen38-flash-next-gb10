# EXL3 decode/verify windows (T16): bound first (2026-10-02)

User: "work on it" (the open EXL3 lever after #212: decode/verify windows still on the grouping kernel).
Rule (memory bound-the-core-before-overheads): bound the core before any overhead work.

Bound, Flash Next EXL3 3.05 bpw, c=1, drafts on (~2.8 tokens a round, ~4–7 verify rows): 40–70 routed pairs a
layer → ~37–60 distinct experts, each read once at minimum: 3 x 2560 x 640 weights at 3 bits = 1.47 MB → 2.6–4.2 GB
a round over 48 layers → 11.7–19 ms at 224 GB/s. In decode an expert has ~1 pair, so there is no cross-row decode
redundancy to remove (unlike prompts); the question is how far the grouped kernel is from the DRAM floor.
Predictions (measured round 38.4 ms on #212):
- Routed experts (group + rot_in + grouped + epilogues) = 15–24 ms a round (40–60 %).
- Effective bandwidth of the grouped kernel at decode shapes 90–150 GB/s → 30–55 % headroom on the experts →
  decode −10…−25 % if it reached ~200 GB/s.
- Below 10 % of the round or above 180 GB/s: no lever here; look elsewhere (dense EXL3 linear, attention, HC).
- **Result (§5bg):** experts ~37 % of a round (in range) but the grouped kernel runs 192–214 GB/s at decode windows
  (out of range high, stop rule): no expert-kernel lever. fp16 matrices 1.34 GB a forward at 203–239 GB/s (tiny GDN
  a/b at 22 GB/s, ~1.7 %). Remaining levers are bytes (fp16 HC matrices → FP8, ~6 %) or forwards, not kernels.

## Second-opinion check (2026-10-02)

Points checked against #212 profiles: `_ple_rows` 0.68 ms of a 5.2 s 8k prefill (dead); decode-once across 64 rows =
#212 (done; decode has ~1 row an expert). Open: `unpack_kernel` (3.1 % of 8k prefill) stores; `prompt_kernel` (32.5 %)
at ~3x its DRAM floor — is the trellis decode (3-bit extraction) its limiter?
- prompt_kernel: DRAM 25–40 % of peak, tensor-pipe instructions < 15 %, issue-bound on INT/ALU (decode) → a cheaper
  3-bit extraction would pay roughly in proportion to its instruction share.
- unpack_kernel: store sectors per request ≥ 2x the ideal (the reviewer's 32 for 16 sectors) and DRAM write < 60 %
  of peak → better stores worth ≤ ~1.5 % of prefill.
- **Result (§5bh):** prompt_kernel memory 21 %, tensor 4 % of instructions, ALU+FMA 67 %, issue 30 % busy at 23 %
  occupancy, 10 cycles/instruction → latency-bound on decode arithmetic (in range). unpack_kernel 160 GB/s, sectors
  written twice, ≤ ~1 % of prefill (in range, small).
