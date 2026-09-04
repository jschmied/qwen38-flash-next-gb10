Same symptom on a different stack, and a data point against the "verification path is inherently different in BF16" reading above (@chenWULUQI): on our box the block-verification path *does* reproduce the q_len=1 path bit-for-bit once three unrelated kernel defects are fixed, so the mismatch is not intrinsic to BF16 verification — at least not on FlashInfer/sm_121.

On a GB10 (sm_121) we serve the same target family (Qwen3.8-27B, NVFP4 body, and Qwen3.8-Flash-Next) with DFlash2 and MTP under vLLM, greedy, and see the output leave the target-only path around token 30 as well. In our case the divergence had three concrete sources, all now fixed or patched upstream, and none of them was the drafter itself:

- the NVFP4 MoE finalize reduced in a non-deterministic order (#54945 / #54948) — Flash-Next only;
- the mamba/GDN align block units seeded from the wrong block size on prefix-cache resume (#54076 / #53798);
- `persistent_topk` on sm_121 returning an order- and set-unstable selection (#55122).

With those fixed, DFlash2 K=1 greedy reproduces the target's run bit-for-bit across restarts on our box (six of six starts, identical draft/accept counts), so on this hardware the "changes at token 30" was never DFlash2's own acceptance logic, and the block forward and the single-token forward agree exactly in BF16. Your FP32 result is consistent with that: FP32 hides an unstable kernel as well as a path difference, so it separates "numerical" from "state" but not "inherent" from "a fixable kernel". Your setup is BF16 on 4×24 GB, so the first source cannot apply and the third only if the QSA top-k runs there; the second applies to any hybrid target with prefix caching on.

Two things that were decisive for us and that you can run without a patched build:

1. **Per-position acceptance and first-divergence index**: `tools/determinism/accept.py` and the agent-loop probe in https://github.com/jschmied/qwen38-flash-next-gb10 read the `/metrics` spec-decode counters per request and log the first token position where draft ≠ target. If the first divergence is always at the same position with `--enforce-eager` and K=1, that points at the target's own forward (batch-shape dependence), not at the drafter.
2. **Logit-level A/B without spec decode**: run the same prompt target-only twice with `--max-num-seqs 1` and then with `--max-num-seqs 2` plus a dummy concurrent request; if the greedy outputs differ, the target is not batch-invariant on your stack and DFlash2 is only exposing it (our finding: `VLLM_BATCH_INVARIANT=1` did not close it on sm_121, a kernel fix did).

Happy to run any specific trace on the GB10 if a repro script lands here.
