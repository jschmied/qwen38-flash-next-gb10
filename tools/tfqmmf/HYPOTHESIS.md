# qmmf prompt tiles (NVFP4 path's block-FP8 dense prompt GEMM), 2026-10-02

8k prefill on #211: `qmmf` 1,085 ms (20.5 %), ~40 TFLOPS vs EXL3 `_gemm` ~68 on the same FLOPs. Decode tiles at prompt
size: 64 x 64 blocks, 4 warps side by side, each warp re-reads all 64 A rows for 16 columns (4 ldmatrix per 8 mma).
Prompt tile (same bits: an element's group sums, block scales and slice order are tile-independent):
- microbench (2,048 rows, Flash Next dense shapes): best config 55–70 TFLOPS (1.4–1.7x), hashes identical to the
  decode tiles in every shape.
- 8k prefill on NVFP4 (#211): −4…−8 %. Below 1.2x in the microbench: stop.
- Microbench result: decode tiles are already 56–64 TFLOPS on the wide (unsplit) shapes; bigger prompt tiles were
  slower except 128x128 on q_proj (+12 %) → dropped. The weak shapes are the split ones (K 6144 → 2560, sk 8:
  22 TFLOPS; 2560 → 512, sk 8: 30). Fusing a tile's slices in one block (same slice order, no cluster): 56 / 49
  TFLOPS, hashes identical.
- Revised prediction, 8k prefill NVFP4 (0.6.2 vs 0.6.2 + fused slices): the clustered launches were ~555 ms of 5.48 s;
  at 2.4x faster → −5…−7 %.
