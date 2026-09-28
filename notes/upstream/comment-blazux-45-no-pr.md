DRAFT — needs the user's go. blazux/qwen3.8-Flash-DGX issue #45 reply (2026-09-28). Go given: "tell him we dont run his container so we cant send a pr".

Thanks for the detailed list, that's a clear bar. We have to pass on the PR, though: we don't run your container. Our box serves bare-metal vLLM main (`1ea7c63f4` plus our overlay), so we can't build your image, run `smoke-test.sh`, or produce numbers on your default configuration. A PR we can't test on your stack wouldn't meet your own criteria.

If you want to try it yourself, everything is public:
- the change: [vllm#58863](https://github.com/vllm-project/vllm/pull/58863), activated by `--use-replayssm` with speculative decoding;
- our measurements and method: [speed-of-light §4t–4u, §5l](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/speed-of-light.md).

Happy to answer questions about it here.
