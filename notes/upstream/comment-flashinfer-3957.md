# Comment for flashinfer#3957 (post AFTER the vLLM issue has a number; replace NNNNN)

Different kernel, same suspect, so a data point rather than a repro of your crash: on the
**CUTLASS** nvfp4 fused MoE (not CuTe DSL), sm_121 (GB10), FlashInfer 0.6.17, via vLLM:

- Default `use_fused_finalize=True`: identical inputs give different outputs across calls at
  M = 52, 3 and 1 rows (identical at M = 55). Not a crash, not an OOB we could see; measured as
  3 distinct top-20 logprob vectors out of 3 identical requests.
- `use_fused_finalize=False`: bit-stable, 3 of 3 identical, in three serving shapes (with/without
  prefix caching, with MTP). End-to-end cost +3.6 % decode.
- Posting because you list the atomic scatter-add finalize among the suspects and this flag is a
  one-line way to take it out of a bisection.
- If you A/B the flag on ≤ 0.6.17 with a populated autotune cache dir, the non-fused runner dies at
  init with `Invalid gemm2 profile id` (the two runners share cache entries; fixed by
  `MoERunner.get_cache_key_extras()` from v0.6.18rc2). Use a fresh cache dir for the comparison.

vLLM side: vllm-project/vllm#NNNNN.
