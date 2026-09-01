#!/usr/bin/env python3
"""Adversarial replay: the three ways a plain capture/replay test gives a false 'clean'.

1. RARE EVENTS. End-to-end divergence is ~100% per request, but a request issues ~1e4-1e5 kernel
   calls. A per-call trip rate of 1e-4 already explains that, and 5 replays would miss it
   entirely. So: many thousands of iterations, and report the observed RATE, not a boolean.

2. PAGE BORDERS / OUT-OF-BOUNDS READS. A kernel reading past its tensor gets live neighbours in
   the server and whatever the allocator left in replay -- so a matching replay proves nothing.
   Positive test instead: hold the tensor bytes fixed and change ONLY the bytes AROUND it. If the
   output moves, the kernel read memory it does not own. This is decisive, not suggestive.

3. MEMORY LAYOUT / ALIGNMENT. clone() takes whatever address the allocator hands out. A kernel
   using vectorised loads can behave differently at different alignments mod 16/128/4096. So the
   same input is replayed at several deliberate offsets.

Usage: python kernelbox_adversarial.py <capture-dir> [iters]
"""
import sys, os, glob, torch

d = sys.argv[1] if len(sys.argv) > 1 else "/opt/llm/kbox"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

files = sorted(glob.glob(os.path.join(d, "*.pt")))
if not files:
    sys.exit(f"no captures in {d}")
print(f"  device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"  {len(files)} capture(s), {ITERS} iterations per rate test\n")

import vllm  # noqa: F401  registers torch.ops.vllm.* / torch.ops._C.*


def clone_any(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(clone_any(i) for i in x)
    if isinstance(x, dict):
        return {k: clone_any(v) for k, v in x.items()}
    return x


def same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def place(t, offset_elems, poison=None, pad_elems=8192):
    """Copy t into the middle of a larger buffer at a chosen offset.
    poison fills everything OUTSIDE t, so out-of-bounds reads see it."""
    t = t.contiguous()
    buf = torch.empty(t.numel() + offset_elems + pad_elems, dtype=t.dtype, device=t.device)
    if poison is not None:
        if buf.is_floating_point():
            buf.fill_(float(poison))
        else:
            buf.fill_(int(poison))
    view = buf[offset_elems:offset_elems + t.numel()].view(t.shape)
    view.copy_(t)
    return view, buf


def run(op, args, kwargs):
    ret = op(*args, **kwargs)
    torch.cuda.synchronize()
    return ret


for f in files:
    c = torch.load(f, weights_only=False)
    key, call = c["op"], c["call"]
    nsname, _, opname = key.partition(".")
    try:
        op = getattr(getattr(torch.ops, nsname), opname)
    except Exception as e:
        print(f"  {key} call={call}: CANNOT RESOLVE ({e})"); continue
    print(f"  === {key} call={call} ===")

    # ---- 1. rate test -------------------------------------------------------
    ref = None
    bad = 0
    first_bad = None
    for i in range(ITERS):
        a, k = clone_any(c["args_pre"]), clone_any(c["kwargs_pre"])
        r = run(op, a, k)
        state = (clone_any(r), clone_any(a))
        if ref is None:
            ref = state
        elif not (same(ref[0], state[0]) and same(ref[1], state[1])):
            bad += 1
            if first_bad is None:
                first_bad = i
    rate = bad / max(1, ITERS - 1)
    print(f"      rate test   : {bad}/{ITERS-1} differ  rate={rate:.2e}"
          + (f"  first at iter {first_bad}" if first_bad is not None else "  (none)"))
    if bad == 0:
        print(f"        -> at this sample size a per-call rate above ~{3.0/ITERS:.1e} is excluded"
              f" (rule of three, 95% CI)")

    # ---- 2. out-of-bounds probe --------------------------------------------
    tensor_idx = [i for i, a in enumerate(c["args_pre"]) if isinstance(a, torch.Tensor)]
    oob_hits = []
    for j in tensor_idx:
        outs = []
        for poison in (0, 1):
            args = list(clone_any(c["args_pre"]))
            try:
                v, _buf = place(args[j], 0, poison=poison)
            except Exception:
                break
            args[j] = v
            try:
                r = run(op, tuple(args), clone_any(c["kwargs_pre"]))
            except Exception as e:
                outs = []; print(f"        arg[{j}] OOB probe skipped ({type(e).__name__})"); break
            outs.append((clone_any(r), clone_any(args)))
        if len(outs) == 2 and not (same(outs[0][0], outs[1][0]) and same(outs[0][1], outs[1][1])):
            oob_hits.append(j)
    print(f"      OOB probe   : "
          + (f"READS OUT OF BOUNDS via arg{oob_hits}" if oob_hits
             else "no dependence on surrounding bytes"))

    # ---- 3. alignment sweep -------------------------------------------------
    align_hits = []
    for j in tensor_idx:
        outs = {}
        for off in (0, 1, 2, 4, 8, 16, 64, 512):
            args = list(clone_any(c["args_pre"]))
            try:
                v, _buf = place(args[j], off, poison=None)
                args[j] = v
                r = run(op, tuple(args), clone_any(c["kwargs_pre"]))
            except Exception:
                continue
            outs[off] = (clone_any(r), clone_any(args))
        if len(outs) >= 2:
            ks = sorted(outs)
            base = outs[ks[0]]
            if any(not (same(base[0], outs[o][0]) and same(base[1], outs[o][1])) for o in ks[1:]):
                align_hits.append(j)
    print(f"      alignment   : "
          + (f"OUTPUT DEPENDS ON ALIGNMENT via arg{align_hits}" if align_hits
             else "stable across offsets 0..512"))
    print()

print("  Reading:")
print("    rate>0            -> the kernel is nondeterministic on its own. Report it.")
print("    OOB probe hit     -> it reads memory it does not own; in the server those bytes are")
print("                         live neighbours, so the value it computes depends on scheduling.")
print("                         This is the 'wrong values' case and explains large jumps.")
print("    alignment hit     -> behaviour varies with address; replay at one address is worthless")
print("                         and the server's allocator decides which path you get.")
print("    all clean at this ITERS -> box exonerated only DOWN TO the rate bound printed above.")
