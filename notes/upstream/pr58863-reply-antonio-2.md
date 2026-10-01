DRAFT — user's go 2026-10-01 ("yes", to the merge + this reply); post only after the merge's tests pass. vllm-project/vllm#58863, reply to @antoniocuegervas.

@antoniocuegervas thanks, this settles both points. Your PIECEWISE columns show no cost from the reworked commit at four streams (115.5 vs 115.0 and 118.0 vs 118.2 ms), so our 141 → 132 tok/s cell was within the ±7 % we see between starts at K=5, not the commit. And FULL graphs taking one stream from 75.3 to 73.0 ms, level with native FULL_DECODE_ONLY, is what the FULL builders were for.

I merged main into the branch (`MERGE_SHA`). The conflicts were with main's new Mamba2 ReplaySSM speculative decoding: `validate_mamba_cached_kernel` now picks RecoverSSM per architecture (Kimi KDA and Qwen4Exp), and every other speculative `--use-replayssm` model takes main's FlashInfer Mamba2 path. TEST_LINE

Written with AI assistance (Claude Code).
