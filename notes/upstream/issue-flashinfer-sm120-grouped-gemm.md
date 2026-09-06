# DRAFT — flashinfer-ai/flashinfer performance issue (do not post without go)

**Title:** SM120/SM121 CUTLASS grouped MoE GEMM (`fused_moe_120`) is latency-bound at one CTA per SM on many-expert
small-N MoEs (Qwen3.8-Flash-Next: 512 experts, N=640, ~150 rows per expert)

**Summary.** On a GB10 (sm_121, 48 SMs, 24 MiB L2) the NVFP4 W4A4 grouped GEMMs of `cutlass_fused_moe` run at
~52 TFLOPS on a 7.5k-token prefill of Qwen3.8-Flash-Next (512 experts, hidden 2560, intermediate 640, top-10), i.e.
~147 rows per expert problem. ncu (`--set full`, FlashInfer 0.6.17, the AOT `fused_moe_120` module) on GEMM1 and GEMM2:

| | GEMM1 (4.19 ms) | GEMM2 (5.61 ms) |
| --- | --- | --- |
| grid / block | 48 CTAs (1 per SM), 384 threads | same |
| registers / dyn. smem per block | 168 / 89 KB → occupancy limit 1 block (regs) and 1 block (smem) | same |
| achieved warps active | 22.9 % | 22.8 % |
| issue slots busy | 12.4 % | 9.6 % |
| "No Eligible" cycles | 87.7 % | ~90 % |
| active / eligible warps per scheduler | 2.59 / 0.16 | – |
| tensor pipe | 28 % | – |
| memory throughput / L2 hit | 32 % / 74 % | 35 % / 70 % |

So the kernel is neither bandwidth- nor compute-bound; each SM holds one CTA whose 12 warps cannot cover the load
latency. Things that do NOT move it (all measured, all bit-identical): every tactic in the SM120 candidate table
(0–31 for GEMM1, 0–63 for GEMM2; all within ±2 %), cold vs warm L2 across replay passes, and the tile-scheduler
`max_swizzle_size` 1/2/4/8 × raster AlongN/AlongM (flat at 7.5k; swizzle ≥ 4 is −14 % at 29k). At decode shapes the
same kernels sit 1.2–1.6× above the expert-byte floor (M=1: 141 µs vs 90; M=4: 530 vs 360; M=256: 6.5 ms vs 4.6).

**Ask.** Is a two-CTA-per-SM variant of the SM120 grouped blockscaled kernel feasible (≤ 84 registers, ≤ 49 KB smem,
e.g. a 128×64 tile with fewer stages that actually lowers register pressure), or a schedule that overlaps the next
problem's prologue with the current mainloop? For 512-expert MoEs the per-problem tile count is tiny (2 × 10 tiles
for GEMM1), so the persistent scheduler switches problems every few tiles.

Repro: `tools/moe_tactics.py` / `tools/moe_ncu.py` in <repo>, `notes/data/pr12.txt`, `pr12c.txt`.
