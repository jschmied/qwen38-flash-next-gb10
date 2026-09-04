Two things measured on our GB10 this week that apply directly to this image, plus one thing not to spend a tournament on.

**1. The blockwise-FP8 GEMM's `M % 4` slow path (the preview image lacks vllm#52775).** On the image's vLLM the sm_12x blockwise-FP8 kernel takes a `swap_ab` route whenever the scheduled chunk's row count is not a multiple of 4 (`(M <= 64) || (M % 4 != 0)`), and that route is ~1.6× slower at prefill M. Upstream fixed it in C++ on 08-19, after the image was cut. Measured on the stock image, prefix cache off, no spec, same prompt ±1 token:

| chunk rows | batch 8192 | batch 4096 |
|---|---|---|
| 7,503 (mod 4 = 3) | 4.52 s | 3.05 s |
| 7,508 (mod 4 = 0) | **2.86 s** | **2.81 s** |
| 29,263 / 29,268 | 12.60 / **11.50 s** | 11.11 / 11.11 s |

Your hybrid mode routes the GDN and QSA side layers through this very kernel, so it is more exposed than the stock NVFP4 checkpoint. Drop-in, no rebuild: `tools/main/fp8_m4pad_patch.py` in our repo pads M to a multiple of 4 inside `apply_block_scaled_mm` (zero rows + unit scale rows, output sliced, opaque custom op so `torch.compile` does not specialise on M); env `VLLM_FP8_PAD_M4=0` turns it off. Same target file as the image's. Server-level validation on our preview venv: M4PAD_NUMBERS.

**2. The trailing prefix-cache block under MTP.** With any EAGLE-family drafter (MTP included) vLLM drops the last full block on every prefix-cache hit, so every agent turn re-prefills it. vllm#53388 (merged 09-01, main only) adds `disable_eagle_block_drop` to the speculative config. Measured on our main-build serve, MTP n=3, prefix caching on, 8-turn EOS-correct agent loop, three interleaved starts:

| | default | flag on |
|---|---|---|
| warm turns (3–8), mean of 18 | 2.05 s | **1.52 s (−26 %)** |
| cached tokens per warm turn | 4,800 | **6,400** |
| MTP acceptance | 53–56 % | 57–60 % |

Acceptance does not move. For this image it is a port of the four `vllm/` files of #53388 (`config/speculative.py`, `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`, `v1/core/single_type_kv_cache_manager.py`); with `MTP=2 PREFIX_CACHE=1` as defaults it is the single largest per-turn win we know of for agent loops on this model.

**3. Not worth testing here: our M-chunking PR (vllm#55180).** It only fires when a weight exceeds the L2 (24 MiB on GB10); Flash-Next's largest FP8 side-layer weights are 25–30 MiB, so the bound is ~3 % and the server-level measurement was null (2.73 vs 2.71 s at 8k, 10.70 vs 10.64 s at 30k). It matters for dense models with 100+ MiB projections, not this one.

Both patches live at https://github.com/jschmied/qwen38-flash-next-gb10/tree/main/tools/main with the measurements in `notes/prefill-investigation.md` (findings 69/70, 89, 94).
