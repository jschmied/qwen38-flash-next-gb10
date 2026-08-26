# Why no source build is needed

PR #53896 touches three non-Python files:

    csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu
    csrc/libtorch_stable/ops.h
    csrc/libtorch_stable/torch_bindings.cpp

The change is a signature extension:

```diff
-    double scale, double norm_eps);
+    double scale, double norm_eps, const std::string& output_gate_activation);
```

```diff
-      "float scale, float norm_eps=1e-5) -> ()");
+      "float scale, float norm_eps=1e-5, "
+      "str output_gate_activation='silu') -> ()");
```

A prebuilt wheel registers the **old** schema, so any call passing the new argument fails.
And the default is wrong for this model: `Qwen/Qwen3.8-Flash-Next` has
`output_gate_type = sigmoid`, so silently accepting the `silu` default would compute the
wrong activation rather than fail loudly.

That looked like a mandatory source build — hours of compilation on a box whose unified
memory pool has been taken down by heavy compiles before.

It is not, because the op is unreachable without speculative decoding:

```python
# vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
:1808   and attn_metadata.num_spec_decodes > 0
:1815   and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
:532    return "torch.ops._C.fused_gdn_decode_post_conv_mtp is not built"
```

It is gated on `num_spec_decodes > 0`, consumes `spec_state_indices_tensor`, and there is
an explicit "is not built" path that degrades rather than crashes. Serving with no
`--speculative-config` never reaches it.

**Consequence:** the port is a Python file overlay (73 non-test files), and speculative
decoding is off the table for this prototype until someone builds the kernel.

The remaining Triton ops the PR adds (hyperconnection, Qwen Sparse Attention) JIT at
runtime and need no build.
