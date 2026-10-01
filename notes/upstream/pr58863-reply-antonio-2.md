POSTED 2026-10-01 as https://github.com/vllm-project/vllm/pull/58863#issuecomment-5928420303 — user's go 2026-10-01 ("yes" to merge + reply), after the merge's tests passed. vllm-project/vllm#58863, reply to @antoniocuegervas.

@antoniocuegervas thanks, this settles both points. Your PIECEWISE columns show no cost from the reworked commit at four streams (115.5 vs 115.0 and 118.0 vs 118.2 ms). That is at K=3; our 141 → 132 tok/s at K=5 was one start per arm, inside the ±7 % we see between starts, so I read it as noise, not the commit. And FULL graphs taking one stream from 75.3 to 73.0 ms, level with native FULL_DECODE_ONLY, is what the FULL builders were for.

I merged main into the branch (`c84caa4739`). The conflicts were with main's new Mamba2 ReplaySSM speculative decoding: `validate_mamba_cached_kernel` now picks RecoverSSM per architecture (Kimi KDA and Qwen4Exp), and every other speculative `--use-replayssm` model takes main's FlashInfer Mamba2 path.

At the merge, on GB10 (main's precompiled kernels for `e5e38ba9b`, no csrc changes since): the GDN and PLE RecoverSSM
kernel tests (73), the RecoverSSM config tests (18), the ReplaySSM/RecoverSSM cases in `tests/test_config.py` (5) and the
Kimi KDA and cudagraph-manager tests (138 passed, 15 skipped) all pass.

Written with AI assistance (Claude Code).
