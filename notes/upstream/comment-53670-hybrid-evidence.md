Evidence from the hybrid case this issue names (GDN + sparse attention + in-checkpoint MTP, mamba-aligned prefix caching), now that #53388 gives a switch:

Qwen3.8-Flash-Next on a single GB10 (sm_121), vLLM main `0.28.1rc1.dev352`, MTP n=3, `--enable-prefix-caching`, batch 4096, 8-turn agent loop (each turn appends ~130 tokens to a ~7.5k-token shared prefix, EOS-correct), three interleaved server starts per arm:

| | default (block dropped) | `disable_eagle_block_drop: true` |
|---|---|---|
| warm turns 3–8, mean of 18 | 2.05 s | 1.52 s (−26 %) |
| prefix-cache hits per warm turn | 4,800 tokens | 6,400 tokens |
| MTP acceptance (3 starts) | 56.1 / 53.3 / 53.8 % | 59.5 / 57.5 / 60.0 % |
| s per turn incl. cold turns (3 starts) | 2.75 / 2.49 / 2.51 | 2.15 / 2.11 / 2.10 |

So on this model the dropped block costs a quarter of every warm turn and keeping it changes acceptance by nothing measurable, which is the trade #53388 warns about coming out clean here. The remaining cold turn in the loop (turn 2 misses on both arms) is the mamba-align "first repetition" behaviour, independent of this issue.
