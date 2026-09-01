#!/usr/bin/env python3
"""Replay captured kernel calls standalone and test the two box properties.

  determinism  same captured inputs, N runs -> outputs bit-identical?
  purity       replay(captured_inputs) == the output captured inside the server?

Runs in a FRESH process with no vLLM engine, no scheduler, no other requests: if a kernel
misbehaves here, the cause is inside that kernel, not in the serving stack around it.

Mutating ops are handled explicitly: every arg is restored from the pre-image before each run,
and the comparison covers BOTH the return value and every arg the kernel wrote into. Comparing
only the return value would miss the topk case entirely, which writes its result into `blocks`.

Usage: python kernelbox_replay.py <capture-dir> [runs]
"""
import sys, os, glob, torch

d = sys.argv[1] if len(sys.argv) > 1 else "/opt/llm/kbox"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

files = sorted(glob.glob(os.path.join(d, "*.pt")))
if not files:
    sys.exit(f"no captures in {d}")
print(f"  device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"  {len(files)} captured call(s) in {d}, {N} replays each\n")

import vllm  # registers torch.ops.vllm.* and torch.ops._C.*  # noqa: F401


def restore(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(restore(i) for i in x)
    if isinstance(x, dict):
        return {k: restore(v) for k, v in x.items()}
    return x


def same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def maxdiff(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape:
        if a.is_floating_point():
            return (a.float() - b.float()).abs().max().item()
        return (a != b).sum().item()
    return float("nan")


for f in files:
    c = torch.load(f, weights_only=False)
    key, call = c["op"], c["call"]
    nsname, _, opname = key.partition(".")
    try:
        op = getattr(getattr(torch.ops, nsname), opname)
    except Exception as e:
        print(f"  {key} call={call}: CANNOT RESOLVE ({e})")
        continue

    runs = []
    for _ in range(N):
        args = restore(c["args_pre"])
        kwargs = restore(c["kwargs_pre"])
        ret = op(*args, **kwargs)
        torch.cuda.synchronize()
        runs.append((restore(ret), restore(args), restore(kwargs)))

    det = all(same(runs[0][0], r[0]) and same(runs[0][1], r[1]) for r in runs[1:])
    pure_ret = same(runs[0][0], c["ret"])
    pure_args = same(runs[0][1], c["args_post"])

    tag = f"{key} call={call}"
    print(f"  {tag}")
    print(f"      determinism ({N} replays)  : {'BIT-IDENTICAL' if det else 'DIFFERS'}")
    if not det:
        for i, r in enumerate(runs[1:], 2):
            if not same(runs[0][1], r[1]):
                for j, (a0, aj) in enumerate(zip(runs[0][1], r[1])):
                    if isinstance(a0, torch.Tensor) and not same(a0, aj):
                        print(f"        run1 vs run{i}: arg[{j}] differs, maxdiff={maxdiff(a0, aj):.3e}")
            if not same(runs[0][0], r[0]):
                print(f"        run1 vs run{i}: return differs, maxdiff={maxdiff(runs[0][0], r[0]):.3e}")
    print(f"      purity vs server return    : {'MATCHES' if pure_ret else 'DIFFERS'}")
    print(f"      purity vs server arg writes: {'MATCHES' if pure_args else 'DIFFERS'}")
    if not pure_args:
        for j, (a, b) in enumerate(zip(runs[0][1], c["args_post"])):
            if isinstance(a, torch.Tensor) and not same(a, b):
                print(f"        arg[{j}] shape={tuple(a.shape)} dtype={a.dtype} maxdiff={maxdiff(a, b):.3e}")
    print()

print("  Reading:")
print("    determinism DIFFERS      -> the kernel itself is nondeterministic; reportable alone.")
print("    determinism BIT-IDENTICAL but purity DIFFERS -> the kernel is deterministic given its")
print("       inputs, but inside the server it read something not in them (stale workspace,")
print("       uninitialised memory, a global). That is the 'wrong values' case.")
print("    both clean               -> this box is exonerated; the divergence enters elsewhere.")
