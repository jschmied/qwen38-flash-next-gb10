A2 FNMTPDENSE4 (speed-of-light 5f queued counterfactual 2), written 2026-09-26 before the run.
Change: MTP drafter fc_embedding, fc_hidden, self_attn.qkv_proj, self_attn.o_proj BF16 -> NVFP4 (draft-head recipe,
_nvfp4_rows_gemv_kernel as custom op fn_nvfp4_linear). Only drafts change; target verifies.
Microbench (L2 flushed, graph replay): per draft step BF16 709 us -> 231 us at M=1, 565 -> 223 us at M=4 (step 1);
per cycle ~-1.30 ms if in-model matches. Weight quant rel-err ~0.095 per layer (random-weight proxy).
H1: per verify cycle (ms/tok x accept_len) -0.9 ... -1.4 ms (-1.7 ... -2.6 %); accept_len within -0.05 ... +0.03;
    ms/tok -1 ... -3 % net. The net ms/tok is the decision metric (acceptance can eat the win).
H0: accept_len drops > 0.08 -> the drafter's attention/input projections are precision-sensitive; try FP8 instead.
Text: may differ from base (acceptance grouping under RecoverSSM, 5b) -> compare timing, not hashes; hashes recorded.
Both arms: VLLM_DISABLE_COMPILE_CACHE=1 (env-gated change inside the compiled drafter graph).
