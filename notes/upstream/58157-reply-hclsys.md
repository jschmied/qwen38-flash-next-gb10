POSTED 2026-09-23. vllm-project/vllm PR #58157, reply to hclsys (2026-09-23).

Thanks for running it on a GB10. Having the test pass on the same hardware as the measurement is useful.

Agreed on the interaction with #57512. For the record, this is what happens on SM12x if both land. With the
defaults (`VLLM_USE_DEEP_GEMM=1`, `VLLM_USE_DEEP_GEMM_E8M0=1`), every float32-scale blockwise-FP8 checkpoint that is
not on the denylist takes the requantizing route. So it gets this warning once per worker process, at every start.
`VLLM_USE_DEEP_GEMM=0` is the only way to quiet it. That is the intended outcome. On this checkpoint the route
costs 2.6 % relative weight error and was 8 % slower than CUTLASS on our GB10, so a per-start warning seems proportionate.
The two PRs are independent and can land in either order.
