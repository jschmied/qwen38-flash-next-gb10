# POSTED 2026-09-06 (user go "post it"): https://github.com/vllm-project/vllm/pull/53899#issuecomment-5560263095

One more finding on this branch, GB10 / sm_121, TP=1, with graphs enabled (the default PIECEWISE, also FULL_DECODE_ONLY):
**every forward consumes the previous step's PLE outputs.** `capture_model()` signals dummy PLE outputs and then runs real
steps through `execute_model()` that submit real requests to the offload worker; the first real wait passes on the dummy
signal, its release resets the flag, and the worker's late copy raises the flag for the next step — the semaphore stays one
step ahead for the life of the server. `cudagraph_mode=NONE` is correct.

How it shows: identical sequential requests give two bit-identical classes (the cold first one, and all later ones), so a
"same prompt twice" check passes; reading the PLE output buffer back after each forward shows exactly the previous step's
rows (0 at the first real step, 32 at a cold 1,460-token request, 1,460 at a cold 1,999-token request); a trace of every
semaphore reset/signal/wait pins the origin to the dummy signal in `capture_model()`.

Fix (11 lines, `PleOffloadConnector.prepare_forward`): reset every layer's semaphore on the model stream before a real
request is launched, so the wait can only be satisfied by this step's copy. PR against this branch: https://github.com/peakcrosser7/vllm/pull/13. With it,
every real step consumes exactly its own rows from the first one (32/16/2/1 at init, then 1,460 ×3, 1,999 ×2; hashes equal to the NONE run's), the cold first request gives the warm logprob (−0.2638), 16 identical requests = 1 class, and the position-resolved set is bit-exact sequentially at 1,460 / 1,999 / 5,960 tokens (0 flips, spread 0.000, 1/8 distinct 64-token completions each); the concurrent batches keep 0 / 416 / 665 flips, identical to the cudagraph-off run — the batch-shape axis, not this defect.
