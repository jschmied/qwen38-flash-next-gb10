# SPDX-License-Identifier: Apache-2.0
"""FNCAP: env-gated tensor capture for per-kernel replay (jschmied 2026-09-24, local tool, not upstream).

FN_CAPTURE_DIR=<dir> turns it on. Run the server eager (--enforce-eager): hooks inside compiled regions may not fire.
- Selected modules (FN_CAPTURE_RE, a regex on qualified names): for each, the module's parameters and buffers once
  (as the live model holds them after load-time processing), plus inputs and outputs of up to FN_CAPTURE_N decode
  calls (rows <= 64) and 3 prefill calls (rows > 64), after skipping FN_CAPTURE_SKIP calls (warm-up and profiling).
- Every router (`*.mlp.gate`): top-10 expert ids of every call, all layers (routing locality at the verify batch).
Timings of a capture run are meaningless (GPU->CPU copies sync); only the data is used.
"""
import json
import os
import re
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_DIR = os.environ.get("FN_CAPTURE_DIR", "")
_RE = re.compile(os.environ.get(
    "FN_CAPTURE_RE",
    r"(layers\.(3|4)\.(linear_attn\.(in_proj_qkv|in_proj_qkvz|in_proj_z|in_proj_ba|out_proj)"
    r"|self_attn\.(q_proj|qkv_proj|o_proj)|mlp\.shared_expert(\.[a-z_]+)?|mlp\.gate|mlp\.experts"
    r"|(attn|mlp)_hyper_connection)$)|(^|\.)lm_head$|mtp\.layers\.0\.(self_attn\.(q_proj|o_proj)|mlp\.experts)$"))
_N = int(os.environ.get("FN_CAPTURE_N", "40"))
_SKIP = int(os.environ.get("FN_CAPTURE_SKIP", "20"))
_lock = threading.Lock()
_route = {}          # layer name -> list of int16 [rows, 10]


def _cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


_BENCH_SET = {}
_BENCH_DONE = [False]
_IN_BENCH = [False]      # set while _bench() runs: hooks must not capture or log routing from bench calls
BW = 220e9


def _param_bytes(m, args, kwargs):
    total = sum(p.numel() * p.element_size() for p in m.state_dict(keep_vars=True).values()
                if isinstance(p, torch.Tensor))
    rl = kwargs.get("router_logits") if isinstance(kwargs, dict) else None
    if rl is None and len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].dim() == 2 \
            and args[1].shape[-1] in (512,):
        rl = args[1]
    if rl is not None and type(m).__name__.lower().find("moe") >= 0:
        e = len(torch.unique(torch.topk(rl.float(), 10, dim=-1).indices))
        return total * e / rl.shape[-1], e
    return total, None


def _bench():
    """Time every captured module on its first decode call's real inputs, L2 flushed before each call."""
    dev = torch.cuda.current_device()
    scratch = torch.empty(64 << 20, dtype=torch.uint8, device=dev)       # 64 MiB > the 24 MiB L2
    res = []
    for name, st in _BENCH_SET.items():
        args, kwargs, m, call = st["gpu"]
        r = {"name": name, "class": type(m).__name__,
             "quant_method": type(getattr(m, "quant_method", None)).__name__}
        try:
            with torch.inference_mode():
                f = (lambda: call(*args, **kwargs)) if call is not None else (lambda: m(*args, **kwargs))
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
                ts = []
                for _ in range(25):
                    scratch.zero_()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); f(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
                ts.sort(); r["eager_us"] = round(ts[len(ts) // 2], 1)
                try:
                    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                    g = torch.cuda.CUDAGraph(); gz = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(s):
                        f(); scratch.zero_()
                        with torch.cuda.graph(g, stream=s):
                            scratch.zero_(); f()
                        with torch.cuda.graph(gz, stream=s):
                            scratch.zero_()
                    torch.cuda.synchronize()
                    def tg(gr):
                        v = []
                        for _ in range(25):
                            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            a.record(); gr.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000)
                        v.sort(); return v[len(v) // 2]
                    r["graph_us"] = round(tg(g) - tg(gz), 1)
                except Exception as ex:
                    r["graph_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                    torch.cuda.synchronize()
            nb, e = _param_bytes(m, args, kwargs)
            rows = next((a.shape[0] for a in list(args) + list((kwargs or {}).values()) if isinstance(a, torch.Tensor)), None)
            r.update(rows=rows, weight_bytes=int(nb), distinct_experts=e, floor_us=round(nb / BW * 1e6, 1))
            best = r.get("graph_us", r["eager_us"])
            r["ratio_to_floor"] = round(best / r["floor_us"], 2) if r["floor_us"] else None
        except Exception as ex:
            r["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        res.append(r)
        logger.warning("FNCAP bench %s", json.dumps(r))
    json.dump(res, open(os.path.join(_DIR, "bench.json"), "w"), indent=1)


def _maybe_bench():
    if _BENCH_DONE[0] or not os.path.exists(os.path.join(_DIR, "BENCH")):
        return
    _BENCH_DONE[0] = True
    _IN_BENCH[0] = True
    try:
        _bench()
    except Exception as ex:                                              # never take the server down
        logger.warning("FNCAP bench failed: %s", ex)
    finally:
        _IN_BENCH[0] = False
    with _lock:
        _flush()
    open(os.path.join(_DIR, "BENCH_DONE"), "w").write("1\n")


def _rows(args):
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() >= 1:
            return a.shape[0]
    return -1


def _save_params(name, mod, d):
    state = {k: v.detach().to("cpu", copy=True) for k, v in mod.state_dict(keep_vars=True).items()
             if isinstance(v, torch.Tensor)}
    meta = {"name": name, "class": type(mod).__name__,
            "quant_method": type(getattr(mod, "quant_method", None)).__name__,
            "attrs": {k: repr(getattr(mod, k)) for k in ("input_size", "output_size", "weight_block_size",
                      "input_size_per_partition", "output_size_per_partition", "top_k", "global_num_experts",
                      "local_num_experts", "hidden_size", "intermediate_size_per_partition") if hasattr(mod, k)},
            "params": {k: [str(v.dtype), list(v.shape)] for k, v in state.items()}}
    torch.save(state, os.path.join(d, "params.pt"))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def _hook_module(name, mod):
    d = os.path.join(_DIR, name.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    st = {"calls": 0, "dec": 0, "pre": 0, "params": False}

    def hook(m, args, kwargs, out):
        if _IN_BENCH[0]:
            return
        _maybe_bench()
        with _lock:
            st["calls"] += 1
            if st["calls"] <= _SKIP:
                return
            rows = _rows(args)
            if rows <= 0 and isinstance(kwargs, dict):
                rows = _rows(list(kwargs.values()))
            kind = "dec" if 0 < rows <= 64 else "pre"
            if (kind == "dec" and st["dec"] >= _N) or (kind == "pre" and st["pre"] >= 3):
                return
            if not st["params"]:
                _save_params(name, m, d); st["params"] = True
            idx = st[kind]; st[kind] += 1
            if kind == "dec" and "gpu" not in st:
                st["gpu"] = (_clone(args), _clone(kwargs), m, None)
                _BENCH_SET[name] = st
            torch.save({"args": _cpu(args), "kwargs": _cpu(kwargs), "out": _cpu(out), "rows": rows,
                        "call": st["calls"]}, os.path.join(d, f"{kind}{idx:03d}.pt"))
    mod.register_forward_hook(hook, with_kwargs=True)


def _hook_router(name, mod):
    def hook(m, args, out):
        if _IN_BENCH[0]:
            return
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.shape[-1] < 10:
            return
        ids = torch.topk(logits.float(), 10, dim=-1).indices.to(torch.int16).cpu()
        with _lock:
            _route.setdefault(name, []).append(ids)
            n = sum(len(v) for v in _route.values())
            if n % 20000 < len(_route):          # periodic flush
                _flush()
    mod.register_forward_hook(hook)


def _flush():
    torch.save({k: v for k, v in _route.items()}, os.path.join(_DIR, "routing.pt"))


def install(model, draft_model=None):
    if not _DIR:
        return
    os.makedirs(_DIR, exist_ok=True)
    hooked, routers = [], 0
    for root_name, root in (("", model), ("draft.", draft_model)):
        if root is None:
            continue
        for name, mod in root.named_modules():
            q = root_name + name
            if name.endswith("mlp.gate"):
                _hook_router(q, mod); routers += 1
            if _RE.search(name):
                _hook_module(q, mod); hooked.append(q)
    for root_name, root in (("", model), ("draft.", draft_model)):
        if root is None:
            continue
        for name, mod in root.named_modules():
            if name.endswith("lm_head") and hasattr(mod, "quant_method") and _RE.search(name):
                w = next((p for p in mod.parameters()), None)
                if w is None:
                    continue
                hidden = w.shape[-1] if w.dim() == 2 else 2560
                x = torch.randn(4, hidden, device=w.device, dtype=torch.bfloat16)
                _BENCH_SET[root_name + name] = {"gpu": ((x,), {}, mod, lambda x, _m=mod: _m.quant_method.apply(_m, x))}
    import atexit
    atexit.register(lambda: _flush())
    json.dump({"hooked": hooked, "routers": routers}, open(os.path.join(_DIR, "installed.json"), "w"), indent=1)
    logger.warning("FNCAP installed: %d modules captured, %d routers logged, dir %s", len(hooked), routers, _DIR)
