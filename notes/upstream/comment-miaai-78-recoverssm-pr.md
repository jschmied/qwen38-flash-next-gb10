POSTED 2026-09-28 (user's go: "offer 58863 to mia and blazux"). MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#78, follow-up on lever 3.

Follow-up on lever 3 (GDN RecoverSSM): it is now an upstream PR, [vllm#58863](https://github.com/vllm-project/vllm/pull/58863). Verify runs from one read-only checkpoint with a small per-token replay record, and the accepted tokens are committed once after sampling. The PLE short conv uses the same plan, and no Mamba block is reserved per draft position. It is switched on with `--use-replayssm` together with speculative decoding. It needs the V2 model runner, the Triton Mamba backend and PP=1.

On one GB10 (MTP K=3, align mode + prefix caching, KV 4 GiB, 2 starts per arm; numbers from the PR):

| | native GDN spec decode | RecoverSSM |
|---|---|---|
| c=4 decode, ms per verify cycle | 105.1–105.4 | 99.6–99.9 (−5.2 %) |
| agent loop, 8 dependent turns | 1.31 s/turn | 1.10 s/turn (−16 %) |
| KV tokens in the same 4 GiB | 75,678 | 103,953 (+37 %) |

Two notes if you try it:
- **Use the PR head, not an earlier copy of the code.** It includes an align-mode fix (credited to Martin Vit, local-inference-lab): when the accepted tokens ended exactly on a block boundary, the state went to the next, possibly unallocated block.
- **On the `v0.30.0` image it is not a clean apply.** One test file conflicts, and two hunks in `qwen_gdn_linear_attn.py` need their context adjusted. I can push a v0.30.0 backport branch if that helps.

Written with AI assistance (Claude Code).
