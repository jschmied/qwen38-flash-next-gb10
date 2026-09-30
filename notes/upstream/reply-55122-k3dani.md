POSTED 2026-09-30 (user's go: "do all important replies"). vllm-project/vllm#55122, reply to @k3dani.

@k3dani thanks for the same-image comparison. Our kernel at 0.99× stock TTFT against the exact `torch.topk` fallback at 1.24–1.25× is the cleanest number this PR has had. Thanks also for stating what it doesn't show about the current head.

On the `VLLM_QSA_DET_TOPK=0` finding: this PR has no environment gate, since it replaces `persistent_topk` itself. The gate you hit belongs to the overlay that loads the deterministic kernel beside stock. We shipped the same bug in our own launcher and found it on 09-24. `os.environ.get("VLLM_QSA_DET_TOPK")` is true for `"0"` (any non-empty string), and `bool(os.environ.get("VLLM_MOE_DET_FINALIZE"))` behaves the same way, so the only way off was to unset the variable. Comparing the value (`== "1"`) fixes both. It also means any earlier A/B whose "stock" arm set the variable to `0` ran the kernel in both arms, so those should be re-read.

Written with AI assistance (Claude Code).
