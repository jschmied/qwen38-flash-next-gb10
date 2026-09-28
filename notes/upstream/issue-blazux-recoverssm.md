POSTED 2026-09-28 (user's go: "offer 58863 to mia and blazux"). blazux/qwen3.8-Flash-DGX new issue.
Title: Offer: GDN RecoverSSM for MTP + prefix caching (vllm#58863), frees KV and cuts per-turn cost

Your default runs MTP with prefix caching, which puts the GDN layers in align mode. There, native speculative decoding snapshots the full recurrent state after every verify token and reserves one Mamba block per draft position. [vllm#58863](https://github.com/vllm-project/vllm/pull/58863) (ours, open) replaces that with RecoverSSM, the protocol Kimi-K3's KDA already uses upstream. Verify runs from one read-only checkpoint with a small per-token replay record, and the accepted tokens are committed once after sampling. The PLE short conv uses the same plan. It is switched on with `--use-replayssm` plus speculative decoding, and needs the V2 model runner, the Triton Mamba backend and PP=1.

Measured on one GB10 (MTP K=3, align + prefix caching, KV 4 GiB, 2 starts per arm; details in the PR):

| | native GDN spec decode | RecoverSSM |
|---|---|---|
| c=1 decode, ms per verify cycle | 55.7–56.7 | 54.4 |
| c=4 decode, ms per verify cycle | 105.1–105.4 | 99.6–99.9 (−5.2 %) |
| agent loop, 8 dependent turns | 1.31 s/turn | 1.10 s/turn (−16 %) |
| KV tokens in the same 4 GiB | 75,678 | 103,953 (+37 %) |

Outputs differ from the native path only by summation order: the state is accumulated per commit rather than per token. A cache-hit replay reproduces the first pass exactly.

Two notes:
- **Use the PR head.** It includes an align-mode fix (credited to Martin Vit, local-inference-lab): when the accepted tokens ended exactly on a block boundary, the state went to the next, possibly unallocated block.
- **On `v0.30.0` it is not a clean apply.** One test file conflicts, and two hunks in `qwen_gdn_linear_attn.py` need their context adjusted. I can push a v0.30.0 backport branch if you'd like to try it in your tournament.

Written with AI assistance (Claude Code).
