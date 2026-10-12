DRAFT (not posted) - TF #721 comment

Thanks for taking #670 on. I checked #721 on one DGX Spark (GB10, driver 580.173.02), Nemotron 3.5 Lightning MLX-4bit. `prompt-cuts` passes with a 9,000-token prompt of my own.

One gap remains. The store rules are now `.{ .lookahead = 1 }`, so `min_prompt` (4,096) applies to Nemotron CUDA again, and a prompt under 4,096 tokens keeps nothing. The default exists because a keep used to cost an extra prompt pass, and commit 2 removes that cost. A short agent loop (12 turns from 1,161 to 4,821 tokens, `--parallel 1`, two starts each, all 24 replies equal across the three builds):

| build | later turns' TTFT (mean) | first turn |
|---|---|---|
| `e7c523eb` | 0.381 s | 0.199 s |
| #721 | 0.386 s | 0.201 s |
| #721 with `.min_prompt = 0` in those rules | 0.134 s | 0.320 s |

The first turn's extra 0.12 s comes from its three keeps in one 1,161-row chunk (875 MiB kept), which still end the chunk at the first. Staging several keeps inside a chunk would remove it, and #722's smaller states would shrink the copies. Every later turn keeps one state and resumes from the previous turn's. GLM's CUDA rules already set `.min_prompt = 0`.

Would you add it to #721, or should I send it as a follow-up once #721 lands?
