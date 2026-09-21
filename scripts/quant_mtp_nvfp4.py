#!/usr/bin/env python3
"""Quantize the Qwen3.8-Flash-Next MTP drafter to NVFP4.

Why: the MTP head is 5.21 GB and 96.6% of that is its MoE experts, which ship BF16 in every
published checkpoint (`mtp.*` is in exclude_modules). Two consequences:
  * ~2.5 GB of resident memory that could be KV cache;
  * vLLM routes the drafter through UnquantizedFusedMoEMethod, whose backend list excludes
    every quantization-only MoE backend -- that is why `--moe-backend flashinfer_b12x` with MTP
    aborts at drafter construction.

Layout: the MTP stores experts FUSED and 3-D (gate_up_proj [E, 2*I, H], down_proj [E, H, I]).
The body stores them SPLIT per expert and 2-D, with NVFP4 companions. We emit the body's layout,
because that is the one vLLM's NVFP4 MoE loader knows:
    experts.{i}.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}

Everything that is not a 2-D expert projection stays BF16: norms, the hyper-connections (closed
lever -- notes/why-the-hyper-connections-do-not-respond.md), gates, conv-like taps.
"""
import argparse, json, os, re, shutil, glob
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/opt/llm/models/qwen38-flash-next-fp8head")
ap.add_argument("--dst", default="/opt/llm/models/qwen38-flash-next-mtpfp4")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

from flashinfer.fp4_quantization import fp4_quantize   # same path the body was built with
SF_VEC = 16

def nvfp4(w: torch.Tensor):
    """[out, in] bf16 -> (packed uint8 [out, in//2], blockscale f8 [out, in//16], global f32)."""
    w32 = w.to(torch.float32)
    amax = w32.abs().amax().clamp(min=1e-8)
    gs = (amax / (448.0 * 6.0)).to(torch.float32)          # modelopt convention: scale_2
    q, sf = fp4_quantize(w.to(torch.bfloat16).to(a.device),
                         global_scale=(1.0 / gs).to(a.device),
                         sf_vec_size=SF_VEC, is_sf_swizzled_layout=False)
    # fp4_quantize hands back the block scales as raw bytes; the checkpoint format (and the
    # loader) wants them typed as e4m3, exactly as the body's experts store them.
    sf = sf.cpu().view(torch.float8_e4m3fn).reshape(w.shape[0], -1)
    return q.cpu(), sf, gs.cpu()

idx = json.load(open(f"{a.src}/model.safetensors.index.json"))
wm = idx["weight_map"]

# input_scale is a CALIBRATED activation scale, not 1.0 -- an uncalibrated one produces
# plausible-looking wrong output (notes/choosing-a-quant-scheme.md). We have no calibration set
# for the drafter, so borrow the body's LAST layer per-expert values: the MTP head consumes the
# final hidden state, so those activation statistics are the closest available. Approximation,
# recorded as such; acceptance length is the measurement that would expose it.
from safetensors import safe_open as _sopen
_last = max(int(m.group(1)) for k in wm
            if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k)))
print(f"  borrowing input_scale from body layer {_last}")
_iscale = {}
for _k, _s in wm.items():
    _m = re.search(rf"layers\.{_last}\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.input_scale$", _k)
    if _m:
        _iscale.setdefault(_s, []).append((_k, int(_m.group(1)), _m.group(2)))
_ISCALE = {}
for _s, items in _iscale.items():
    with _sopen(f"{a.src}/{_s}", framework="pt") as _f:
        for _k, _e, _p in items:
            _ISCALE[(_e, _p + "_proj")] = _f.get_tensor(_k).clone()
print(f"  loaded {len(_ISCALE)} calibrated input_scale values")
mtp_shards = sorted({s for k, s in wm.items() if re.search(r"(^|\.)mtp\.", k)})
print(f"  mtp lives in {len(mtp_shards)} shard(s): {mtp_shards}")
if a.dry_run:
    raise SystemExit("  dry run: no files written")

os.makedirs(a.dst, exist_ok=True)
new_map = dict(wm)
n_q = 0
b_old = b_new = 0

for sh in sorted({s for s in wm.values()}):
    src = f"{a.src}/{sh}"
    if sh not in mtp_shards:                       # untouched: hardlink, no copy
        dst = f"{a.dst}/{sh}"
        if not os.path.exists(dst):
            try: os.link(src, dst)
            except OSError: shutil.copy(src, dst)
        continue
    t = load_file(src)
    out = {}
    for k, v in t.items():
        b_old += v.numel() * v.element_size()
        m = re.match(r"(.*mtp\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$", k)
        if m and v.dim() == 3:
            base, which = m.group(1), m.group(2)
            E = v.shape[0]
            for e in range(E):
                mat = v[e]                                   # [2I, H] or [H, I]
                parts = ([("gate_proj", mat[: mat.shape[0] // 2]),
                          ("up_proj",   mat[mat.shape[0] // 2:])]
                         if which == "gate_up_proj" else [("down_proj", mat)])
                for name, w in parts:
                    q, sf, gs = nvfp4(w)
                    p = f"{base}.{e}.{name}"
                    out[f"{p}.weight"] = q
                    out[f"{p}.weight_scale"] = sf
                    out[f"{p}.weight_scale_2"] = gs
                    out[f"{p}.input_scale"] = _ISCALE.get(
                        (e, name), torch.tensor(0.00202288, dtype=torch.float32))
                    n_q += 1
            del new_map[k]
            if e % 128 == 0:
                print(f"    {which}: {E} experts done", flush=True)
        else:
            out[k] = v
    for k in out:
        new_map[k] = sh
    for v in out.values():
        b_new += v.numel() * v.element_size()
    save_file(out, f"{a.dst}/{sh}", metadata={"format": "pt"})
    print(f"  {sh}: {len(t)} -> {len(out)} tensors", flush=True)

json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_map},
          open(f"{a.dst}/model.safetensors.index.json", "w"), indent=1)

for f in os.listdir(a.src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy(f"{a.src}/{f}", f"{a.dst}/{f}")

# hf_quant_config: mtp is no longer excluded, and its experts are NVFP4
hq = json.load(open(f"{a.src}/hf_quant_config.json"))
q = hq.get("quantization", hq)
q["exclude_modules"] = [e for e in q.get("exclude_modules", []) if e not in ("mtp.*", "model.mtp.*")]
ql = q.setdefault("quantized_layers", {})
# The VALUE is a dict -- modelopt.py does quantized_layers[k]["quant_algo"] -- and the KEY must be
# the RUNTIME prefix. vLLM remaps the drafter to layers.<num_hidden_layers>, and
# _quantized_layer_prefix_candidates() only swaps language_model.model. <-> model.language_model.,
# so "mtp.layers.0..." never matches. Write both index forms and both prefix forms.
_nl = json.load(open(f"{a.src}/config.json"))
_nl = (_nl.get("text_config") or _nl).get("num_hidden_layers") or _nl.get("num_hidden_layers")
_entry = {"quant_algo": "NVFP4", "group_size": SF_VEC}
for _i in (_nl, 0):
    ql[f"mtp.layers.{_i}.mlp.experts"] = dict(_entry)
    ql[f"model.mtp.layers.{_i}.mlp.experts"] = dict(_entry)
print(f"  quantized_layers: mtp runtime layer index {_nl}")
json.dump(hq, open(f"{a.dst}/hf_quant_config.json", "w"), indent=1)

# THE FILE THAT ACTUALLY MATTERS: vLLM reads config.json["quantization_config"], not
# hf_quant_config.json. Editing only the latter leaves the drafter unquantized and the load dies
# with "Layer mtp.layers.<N>.mlp.experts has no parameter 'w2_input_scale'". Keep both in sync.
_cfg = f"{a.dst}/config.json"
_c = json.load(open(_cfg))
_qc = _c.get("quantization_config")
if _qc is not None:
    _inner = _qc.get("quantization", _qc)
    _inner["exclude_modules"] = [e for e in _inner.get("exclude_modules", [])
                                 if e not in ("mtp.*", "model.mtp.*")]
    # AND the compressed-tensors-style "ignore" list, which is merged into exclude_modules.
    # This is where the real mtp exclusion lives; exclude_modules here is often empty, so
    # clearing only that leaves the drafter excluded and it loads unquantized.
    if "ignore" in _qc:
        _qc["ignore"] = [e for e in _qc["ignore"] if e not in ("mtp.*", "model.mtp.*")]
    _cql = _inner.setdefault("quantized_layers", {})
    for _i in (_nl, 0):
        _cql[f"mtp.layers.{_i}.mlp.experts"] = dict(_entry)
        _cql[f"model.mtp.layers.{_i}.mlp.experts"] = dict(_entry)
    json.dump(_c, open(_cfg, "w"), indent=1)
    print(f"  config.json quantization_config updated ({len(_cql)} quantized_layers)")

print(f"\n  quantized {n_q} expert projections")
print(f"  rewritten shards: {b_old/1e9:.2f} GB -> {b_new/1e9:.2f} GB ({(b_new-b_old)/b_old*100:+.1f}%)")
