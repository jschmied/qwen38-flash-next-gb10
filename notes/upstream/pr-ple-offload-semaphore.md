# OPENED 2026-09-06 (user go "create pr"): https://github.com/peakcrosser7/vllm/pull/13 — body below is the live text
# Branch: jschmied/vllm:ple-offload-wait-fix (1 commit on top of 357e0544, signed off). Open on go, after the validation run.

## Purpose

With CUDA graphs enabled (`cudagraph_mode` PIECEWISE, FULL_DECODE_ONLY, …) every forward of Qwen3.8-Flash-Next with
`VLLM_PLE_CPU_OFFLOAD=1` consumes the **previous step's** PLE outputs. `capture_model()` signals dummy PLE outputs and then
runs real steps through `execute_model()` that submit real requests to the offload worker. The first real wait passes on the
dummy signal (buffer still zero), its release resets the flag, and the worker's copy for that step raises the flag for the
*next* step — the semaphore is one step ahead from then on. `cudagraph_mode=NONE` never hits this (no dummy signal outside
`execute_model`) and is correct.

Symptom that led here: identical sequential requests fall into two classes (cold first request vs all later ones,
bit-identical among themselves), so the defect is invisible to "same prompt twice" checks and only shows when consecutive
steps differ — i.e. always, in real serving (position-resolved logprobs: 758/1,459 top-1 flips at 1,460 tokens between
a request that followed a different request and one that followed an identical one).

## Fix

Reset every layer's semaphore on the model stream before a real request is launched (`PleOffloadConnector.prepare_forward`),
so the GPU-side `ple_offload_wait` can only be satisfied by this step's copy. Per rank, before the `tp_rank` check in
`_launch`, because each TP rank owns its buffer and semaphore. 11 lines, no protocol change for the worker (it already waits
for the reset before copying).

## Evidence (GB10 / sm_121, TP=1, FP8 PLE shards, no spec, prefix cache off)

- PLE output buffer read back after each real forward (hash + non-zero rows), PIECEWISE: first real step 0 rows, cold
  1,460-token request 32 rows (= previous step), next identical request 1,460 rows, cold 1,999-token request exactly 1,460
  rows. NONE: every step exactly its own rows; the NONE hash equals PIECEWISE's *warm* hash.
- Semaphore trace (every reset/signal/wait with caller, both processes): `signal via signal_dummy_outputs <- capture_model`,
  then `execute_model` steps at 32/16/2/1 tokens with worker requests; the 32-token wait sees flag=1 and 0 rows; the
  16-token wait blocks on 0 and is released by the worker's signal *for the 32-token step*; after that every release is
  immediately followed by the previous step's late signal.
- With the reset: every real step consumes exactly its own rows from the first one (32/16/2/1 at init, then 1,460 ×3, 1,999 ×2; hashes equal to the NONE run's), the cold first request gives the warm logprob (−0.2638), 16 identical requests = 1 class, and the position-resolved set is bit-exact sequentially at 1,460 / 1,999 / 5,960 tokens (0 flips, spread 0.000, 1/8 distinct 64-token completions each); the concurrent batches keep 0 / 416 / 665 flips, identical to the cudagraph-off run — the batch-shape axis, not this defect.

## Test plan

Served the model with `cudagraph_mode=PIECEWISE` (default) and NONE; per-step PLE buffer probe; 16 identical sequential
requests (full prompt-logprob vector hashed); position-resolved logprob comparison on 1,460 / 1,999 / 5,960-token prompts.

_Written with AI assistance (Claude Code); every line reviewed by the author._
