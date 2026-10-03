POSTED 2026-10-03 (user: "ok"). TensorFold PR #211 reply to SvangenStudios (2026-10-03).

@SvangenStudios thanks for testing this and finding the bug. Your diagnosis was right, and the fix is your line: [5ad55f9](https://github.com/jschmied/TensorFold/commit/5ad55f92d6848ca39c275ab4bf12f381d0d696f5). Our checkpoint stores the MTP layer's experts with input scales, so it never reached that check.

The commit also adds a test: the tiny checkpoint writer can now store MTP experts as per-expert NVFP4 without input scales, like your export. On that checkpoint, under `--precision checkpoint`, the main layers run FP4 x FP4 and the MTP experts keep `nvfp4_expert_kernel`. The test failed with your startup error before the fix. With it, `test_flashnext_nvfp4_loader.py` and `test_nvfp4_experts_ck.py` pass, 48 tests on one GB10.
