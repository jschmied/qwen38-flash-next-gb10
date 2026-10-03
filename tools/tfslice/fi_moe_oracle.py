#!/usr/bin/env python3
"""Reference for TF's prompt-path experts: FlashInfer's CUTLASS NVFP4 fused MoE (what vLLM runs on GB10) on Flash Next's
expert shapes at TF's piece sizes, autotuned, with the deterministic finalize (no atomics) and the fused one; plus
whether a row's output bits depend on the batch (rows of a 2,048 batch vs the same rows inside 8,192).
Run with a venv that has vllm (scaled_fp4_quant) and flashinfer."""
import json, torch
from vllm import _custom_ops as ops
from flashinfer.fused_moe.core import ActivationType, cutlass_fused_moe
from flashinfer.autotuner import autotune

dev = "cuda"; torch.manual_seed(0); E, H, I, TOPK = 512, 2560, 640, 10
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8)
w13_s = (torch.rand(E, 2 * I, H // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
w2_s = (torch.rand(E, H, I // 16, device=dev) * 0.5 + 0.5).to(torch.float8_e4m3fn)
g1 = torch.ones(E, device=dev); g2 = torch.ones(E, device=dev); a1g = torch.ones((), device=dev); a2g = torch.ones((), device=dev)
X = torch.randn(8192, H, device=dev, dtype=torch.bfloat16)
IDS = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(8192)]).to(torch.int32)
TW = torch.softmax(torch.randn(8192, TOPK, device=dev), dim=-1).float()


def call(m, fused, out):
    a_fp4, a_sf = ops.scaled_fp4_quant(X[:m].contiguous(), a1g)
    cutlass_fused_moe(input=a_fp4, token_selected_experts=IDS[:m].contiguous(), token_final_scales=TW[:m].contiguous(),
                      fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                      output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2],
                      input_sf=a_sf, output=out, tune_max_num_tokens=8192, activation_type=ActivationType.Swiglu,
                      use_fused_finalize=fused)
    return out


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


res = {}
for fused in (False, True):
    for m in (2048, 4096, 8192):
        out = torch.empty(m, H, device=dev, dtype=torch.bfloat16)
        with autotune(True):
            call(m, fused, out); torch.cuda.synchronize()
        ms = timed(lambda: call(m, fused, out))
        res[(fused, m)] = out.clone()
        flop = 2 * m * TOPK * (H * 2 * I + I * H)
        print(json.dumps({"fused_finalize": fused, "rows": m, "ms_per_layer": round(ms, 2), "TFLOPS": round(flop / ms / 1e9, 1),
                          "8k_prompt_48_layers_s": round(ms * 48 * 8192 / m / 1000, 3)}), flush=True)
    a, b = res[(fused, 2048)], res[(fused, 8192)][:2048]
    again = call(8192, fused, torch.empty(8192, H, device=dev, dtype=torch.bfloat16))[:2048]
    print(json.dumps({"fused_finalize": fused, "rows_2048_alone_eq_inside_8192": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
                      "8192_run_to_run_eq": bool(torch.equal(b.view(torch.int16), again.view(torch.int16)))}), flush=True)
