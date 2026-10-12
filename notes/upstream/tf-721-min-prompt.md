POSTED 2026-10-12 as TF #721 comment, short version (user: "short note"). https://github.com/ashhart/TensorFold/pull/721#issuecomment-6120014673

Thanks for taking #670 on. `prompt-cuts` passes here too (GB10, a 9,000-token prompt of my own).

One gap: with the rules now `.{ .lookahead = 1 }`, the default `min_prompt` (4,096) applies again, so prompts under 4,096 tokens keep nothing. Commit 2 removes the extra pass that the default guards against. Adding `.min_prompt = 0` (as GLM's CUDA rules have) cuts later turns' TTFT in a 12-turn agent loop of 1.2k to 4.8k tokens from 0.386 s to 0.134 s, with all replies equal. The first turn costs 0.12 s more, from its three keeps in one chunk. Worth adding here, or a follow-up after this lands?

--- long draft (not posted):
Thanks for taking #670 on. I checked #721 on one DGX Spark (GB10, driver 580.173.02), Nemotron 3.5 Lightning MLX-4bit. `prompt-cuts` passes with a 9,000-token prompt of my own.

One gap remains. The store rules are now `.{ .lookahead = 1 }`, so `min_prompt` (4,096) applies to Nemotron CUDA again, and a prompt under 4,096 tokens keeps nothing. The default exists because a keep used to cost an extra prompt pass, and commit 2 removes that cost. A short agent loop (12 turns from 1,161 to 4,821 tokens, `--parallel 1`, two starts each, all 24 replies equal across the three builds):

| build | later turns' TTFT (mean) | first turn |
|---|---|---|
| `e7c523eb` | 0.381 s | 0.199 s |
| #721 | 0.386 s | 0.201 s |
| #721 with `.min_prompt = 0` in those rules | 0.134 s | 0.320 s |

The first turn's extra 0.12 s comes from its three keeps in one 1,161-row chunk (875 MiB kept), which still end the chunk at the first. Staging several keeps inside a chunk would remove it, and #722's smaller states would shrink the copies. Every later turn keeps one state and resumes from the previous turn's. GLM's CUDA rules already set `.min_prompt = 0`.

Would you add it to #721, or should I send it as a follow-up once #721 lands?
