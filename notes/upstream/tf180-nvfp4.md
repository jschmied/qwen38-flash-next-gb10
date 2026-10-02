DRAFT — needs the user's go. ashhart/TensorFold PR #180 comment (2026-10-02).

@SvangenStudios thanks for the NVFP4 run. On our NVFP4 export (ModelOpt, NVFP4 experts, block-FP8 dense; one GB10) the
recaptures do happen on cb5101d and #180 removes them, two rounds, greedy replies identical:

| ms/token | cb5101d | #180 |
|---|---|---|
| `--parallel 1` engine, one request | 15.93 / 15.90 | 16.05 / 15.87 |
| `--parallel 4` engine, one request | 21.29 / 20.86, 21 captures | 15.95 / 15.95, 0 |
| four different lone requests in a row | 20.9–31.1, 26–29 captures each | 13.8–18.5, 0 |

Our one-request speed is 63 tok/s against your 36, at every setting, so something else may bound your decode. Your
startup lines would tell: "N decode graphs captured" and whether the mapped n-gram tables fit ("do not fit ... page them
from disk").

Written with AI assistance (Claude Code).
