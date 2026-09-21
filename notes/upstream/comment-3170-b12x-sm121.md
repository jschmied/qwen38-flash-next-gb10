DRAFT — needs the user's go. flashinfer-ai/flashinfer issue #3170 (2026-09-21).

Data for action item 2 (validate b12x MoE on SM121), plus a note that its stated blocker is stale.

**Box**: NVIDIA DGX Spark, GB10, `sm_121`, aarch64, flashinfer `0.6.18.post1`, torch `2.13.0+cu130`,
driver 580.x. Tests taken from the `v0.6.18.post1` tag.

`tests/moe/test_b12x_fused_moe.py` passes on SM121:

```
189 passed, 2 warnings in 151.63s   (cold, JIT compile)
189 passed, 2 warnings in  44.61s   (warm cache, second run)
```

This includes the numerical-accuracy tests, not only the structural ones — `test_relu2_micro_accuracy`,
`test_relu2_functional_accuracy`, `test_relu2_wrapper_accuracy`, and the W4A16 variants
`test_relu2_w4a16_direct_micro_accuracy` / `test_relu2_w4a16_functional_accuracy`.

**The `@not_sm121` decorator named in item 2 no longer exists.** It is absent from
`tests/moe/test_b12x_fused_moe.py` both at `v0.6.18.post1` and on `main`, and a repository code
search for `not_sm121` returns no hits. So the item's action ("remove `@not_sm121` from tests") looks
already done or renamed; the validation half is what this comment supplies.

**Scope, stated plainly**: 11 distinct test functions, 48 of the 189 from one activation class. This
says the b12x MoE path runs and is numerically correct on SM121 for what that file covers. It is not
a claim about b12x MoE coverage in general, and I have not run a performance comparison.

Item 3 (b12x FP4 GEMM) I could not run the same way: there is no `test_b12x_*` file under
`tests/gemm`, so if you can name the intended test target I will run it on this box.

_AI assistance (Claude Code) was used in preparing this comment; the runs are from the box described._
