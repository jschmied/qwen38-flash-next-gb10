# OPENED 2026-09-06 (user go "create pr"): https://github.com/peakcrosser7/vllm/pull/13 — body below is the live text

## Purpose

Fix a one-step-behind read of the PLE CPU-offload outputs whenever CUDA graphs are enabled (`cudagraph_mode` PIECEWISE, FULL_DECODE_ONLY, …) on this branch.

`capture_model()` signals dummy PLE outputs and then runs real steps through `execute_model()` that submit real requests to the offload worker. The first real wait passes on the dummy signal (the buffer is still zero), its release resets the flag, and the worker's copy for that step raises the flag for the *next* step. From then on every forward consumes the **previous step's** per-layer embeddings; only identical consecutive requests hide it. `cudagraph_mode=NONE` never signals dummy outputs outside `execute_model` and is correct.

Fix: reset every layer's semaphore on the model stream before a real request is launched (`PleOffloadConnector.prepare_forward`), so the GPU-side `ple_offload_wait` can only be satisfied by this step's copy. Per rank, before the `tp_rank` check in `_launch`, because each TP rank owns its buffer and semaphore. 11 lines; the worker already waits for the reset before copying, so its protocol is unchanged.

## Test Plan

GB10 (sm_121, TP=1), FP8 PLE shards, `VLLM_PLE_CPU_OFFLOAD=1`, no speculation, prefix cache off, `cudagraph_mode=PIECEWISE` (default) and `NONE` as the reference:

1. Read back every PLE layer's GPU output buffer after each real forward (hash + non-zero row count), unfixed vs fixed.
2. 16 identical sequential chat requests with `prompt_logprobs=5`, full per-position vector hashed and grouped into classes.
3. Position-resolved comparison of 8 sequential + 8 concurrent identical requests on 1,460 / 1,999 / 5,960-token prompts (first divergent position, top-1 flips, logprob spread), plus 8 greedy 64-token completions per prompt.
4. A trace of every semaphore reset/signal/wait with caller, in both processes, to locate the unmatched signal.

## Test Result

Unfixed, PIECEWISE: first real step 0 non-zero rows; a cold 1,460-token request 32 rows (= the previous step); the next identical request 1,460 rows; a cold 1,999-token request exactly 1,460 rows. 16 identical requests fall into two classes (cold first, then 15 bit-identical). Trace: `signal via signal_dummy_outputs <- capture_model`, then `execute_model` steps at 32/16/2/1 tokens with worker requests; the 32-token wait sees flag=1 and 0 rows; the 16-token wait blocks and is released by the worker's signal *for the 32-token step*; every later release is followed by the previous step's late signal.

`cudagraph_mode=NONE`: every step exactly its own rows; the NONE buffer hash equals PIECEWISE's *warm* hash, i.e. the warm class is the correct computation.

Fixed, PIECEWISE: every real step consumes exactly its own rows from the first one (32/16/2/1 at init, then 1,460 ×3, 1,999 ×2), hashes equal to the NONE run's; the cold first request gives the same first-token logprob as the warm ones (−0.2638); 16 identical requests = **one** class; the position-resolved set is bit-exact sequentially at 1,460 / 1,999 / 5,960 tokens (0 flips, spread 0.000, 1/8 distinct completions each). The concurrent batches keep 0 / 416 / 665 flips, identical to the NONE run — the batch-shape axis, unrelated to this fix.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>

_This PR includes AI-generated code (Claude Code); every changed line was reviewed and the behavior validated end-to-end by the author._

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
