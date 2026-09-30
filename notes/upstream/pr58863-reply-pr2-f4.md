POSTED 2026-09-30 as https://github.com/vllm-project/vllm/pull/58863#issuecomment-5904899697 — user's go 2026-09-30 ("interrupt work, do both now"). vllm-project/vllm#58863, reply to @lucifer1004 and @stevededrick after merging jschmied/vllm#2 and pushing the FULL-graph commit.

@lucifer1004 thanks, your three commits are in the branch (jschmied/vllm#2, merged as you wrote them), with the FULL-graph builders on top (`3aa3d5b586`) and one fix to the split commit (`3388ba1a25`).

The fix: with one fold per program, the final-state and boundary-state folds ran as programs of the same launch, and in align mode a boundary state can be written into the window's source slot. A crossing window's boundary state goes into its source block, and a window ending exactly on a boundary keeps its final state there too, so a program could read a source tile another program had already overwritten. On our GB10, `test_recoverssm_gdn.py` failed in 9 of 12 runs with the split kernel and 0 of 12 with the previous fused one (first seen as `test_align_boundary_and_final_state_match_native[3-2-4]`). The final folds now run first and the boundary folds as a second launch, which skips windows whose boundary slot is their final slot. Every program still runs one fold. It passes 20 of 20 repeated runs, and the five test files pass 121/121. Could you re-check the commit timing on H200? It is now two launches instead of one.

Your `max_num_seqs <= num_blocks` note is in the description, and the fused CUDA verify fits well as a follow-up once this lands.

@stevededrick thanks for the KDA confirmation and the deterministic boundary repro: 12/48 → 0/48 is the cleanest evidence the column fix has had.

GB10 (sm_121, TP1, Qwen3.8-Flash-Next NVFP4, MTP K=5, FULL_AND_PIECEWISE, KV 4 GiB, one start per arm):

| | before #2 (`5567cc1b25`) | #2 as submitted | head `3388ba1a25` |
|---|---|---|---|
| greedy hashes (code / prose) | reference | identical | identical |
| code c=1, ms/tok | 14.32 | 14.41 | 14.24 |
| prose c=1, ms/tok | 23.03 | 23.26 | 23.26 |
| code c=4, tok/s | 141.3 | 131.4 | 132.5 |
| TTFT 8k / 30k, s | 2.72 / 9.69 | 2.74 / 9.73 | 2.69 / 9.66 |
| MemAvailable at ready | 33.3 GiB | 33.4 GiB | 32.1 GiB |

c=1 and TTFT are level. The c=4 cell spreads about ±7 % between starts on this box, so one start per arm does not settle it. The teardown fix frees nothing measurable here, because we set the KV size explicitly (`--kv-cache-memory-bytes`); the OOM you hit needs the default profiling path.

Written with AI assistance (Claude Code).
