POSTED on #212 (2026-10-02, user: "short comment on 212").

Reproduced on a second GB10, Flash Next EXL3 3.05 bpw, three alternating rounds: 8k prefill 11.4 → 5.3 s (−53.5…−53.8 %),
32k 46.2–46.9 → 21.3 s (−53.8…−54.6 %), the first 16 tokens identical to v0.6.1, decode with drafts level (71.9 vs
71.8–72.0 tok/s). My grouping commits stacked on top add nothing in decode, so I closed #184, #191, #193, #195 and #207.
[Data](https://github.com/jschmied/qwen38-flash-next-gb10/tree/63b995d2c939e7500a8b2ea7b8499fb172d0ddec/notes/data/tfexl3/pr212)

Written with AI assistance (Claude Code).
