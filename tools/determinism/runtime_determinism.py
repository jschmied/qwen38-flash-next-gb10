#!/usr/bin/env python3
"""Is the NON-DETERMINISM below the framework? No vLLM, no model, no FlashInfer.

If llama.cpp, SGLang and vLLM all show it, either (a) something under all three -- CUDA, the
driver, sm_121 -- is non-deterministic, or (b) three teams independently wrote non-deterministic
kernels for the same unusual architecture. This distinguishes them.

Each test repeats one operation N times on IDENTICAL inputs and compares bitwise. Anything that
differs is non-deterministic at the runtime level, beneath any serving framework.

REQUIRES AN IDLE GPU -- it allocates and times on the device.
"""
import torch, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dev = "cuda"
torch.manual_seed(0)
print(f"  device : {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print(f"  torch  : {torch.__version__}  cuda={torch.version.cuda}")

def check(name, fn, *, tol_note=""):
    outs = []
    for _ in range(N):
        o = fn()
        torch.cuda.synchronize()
        outs.append(o.detach().clone())
    ident = all(torch.equal(outs[0], o) for o in outs[1:])
    if ident:
        print(f"  {name:<34} BIT-IDENTICAL over {N} runs")
    else:
        md = max((outs[0] - o).abs().max().item() for o in outs[1:])
        # does it move an argmax? that is what actually changes a token
        am = [o.argmax(dim=-1) for o in outs]
        arg = all(torch.equal(am[0], a) for a in am[1:])
        print(f"  {name:<34} DIFFERS  max|d|={md:.3e}  argmax_stable={arg}{tol_note}")

# 1. plain GEMM, the shape class our dense projections use
A = torch.randn(4096, 2560, device=dev, dtype=torch.bfloat16)
B = torch.randn(2560, 10240, device=dev, dtype=torch.bfloat16)
check("bf16 matmul 4096x2560x10240", lambda: A @ B)

# 2. skinny GEMM, the decode-shaped case (batch 1)
a = torch.randn(1, 2560, device=dev, dtype=torch.bfloat16)
check("bf16 matmul 1x2560x10240 (decode)", lambda: a @ B)

# 3. reduction with a known atomic/split-K flavour
C = torch.randn(8192, 8192, device=dev, dtype=torch.float32)
check("fp32 sum over 8192x8192", lambda: C.sum())

# 4. softmax+matmul, an attention-shaped composite
q = torch.randn(1, 24, 1, 256, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, 24, 4096, 256, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, 24, 4096, 256, device=dev, dtype=torch.bfloat16)
check("sdpa 24h x 4096ctx", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

# 5. topk with deliberate ties -- the pattern at the centre of vllm#54521
t = torch.zeros(512, 8192, device=dev, dtype=torch.float32)
t[:, ::7] = 1.0            # many exactly-equal candidates
check("topk k=512 with ties", lambda: torch.topk(t, 512, dim=1).indices,
      tol_note="  <- ties are EXPECTED to be arbitrary; only instability across runs matters")

# ---------------------------------------------------------------------------
# 6. The same ops, but with ALLOCATOR CHURN between runs.
#
# Tests 1-5 repeat an op on identical allocations, so a kernel that reads
# uninitialised or stale memory gets the SAME stale bytes every time and looks
# clean. Our observed divergence (top-token probability 0.85 -> 0.68) is far too
# large for reduction-order noise, and reading undefined memory is the failure
# class that produces jumps that size. Forcing each run to land at a different
# address is what separates the two.
#
# A DIFFERS here while the matching test above was BIT-IDENTICAL is the
# signature: the arithmetic is fine, the memory it reads is not.
print("\n  -- with allocator churn between runs (different addresses each time) --")

def check_churn(name, mk_inputs, fn):
    outs = []
    junk = []
    for i in range(N):
        # vary the heap so the next allocation cannot reuse the same block
        junk.append(torch.empty(((i * 7919) % 4096 + 1) * 1024, device="cuda"))
        if i % 2:
            junk.clear()
        a = mk_inputs()
        torch.cuda.synchronize()
        outs.append(fn(*a).clone())
        torch.cuda.synchronize()
    del junk
    ident = all(torch.equal(outs[0], o) for o in outs[1:])
    if ident:
        print(f"  {name:<34} BIT-IDENTICAL over {N} runs")
    else:
        md = max((outs[0] - o).abs().max().item() for o in outs[1:])
        am = [o.argmax(dim=-1) for o in outs]
        arg = all(torch.equal(am[0], a) for a in am[1:])
        print(f"  {name:<34} DIFFERS  max|d|={md:.3e}  argmax_stable={arg}")

def _mk_gemm():
    torch.manual_seed(0)
    return (torch.randn(4096, 2560, device="cuda", dtype=torch.bfloat16),
            torch.randn(2560, 10240, device="cuda", dtype=torch.bfloat16))

def _mk_skinny():
    torch.manual_seed(0)
    return (torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16),
            torch.randn(2560, 10240, device="cuda", dtype=torch.bfloat16))

check_churn("bf16 GEMM 4096x2560x10240", _mk_gemm, lambda a, b: a @ b)
check_churn("bf16 skinny GEMM 1x2560x10240", _mk_skinny, lambda a, b: a @ b)

print("\n  Reading: BIT-IDENTICAL everywhere means the runtime is not the source and the")
print("  divergence is above it, in vLLM's own kernels or model code. Any DIFFERS -- and")
print("  especially a churn test differing where its fixed-address twin did not -- points")
print("  beneath the framework, at CUDA/driver/FlashInfer, and is reportable on its own.")
