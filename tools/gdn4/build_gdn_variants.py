"""Build two variants of qwen38-flash-next-mtpfp4 that differ ONLY in the 36 GDN projections (in_proj_qkv, in_proj_z,
out_proj of the target model):
  gdn4    : NVFP4 W4A16 (modelopt layout: uint8 weight [N,K/2], fp8 weight_scale [N,K/16], fp32 weight_scale_2 =
            amax/2688; in_proj_qkv and in_proj_z share one global scale because vLLM fuses them), codes rounded against
            the effective fp8-rounded group scale; quantized from the BF16 originals (RadixArk NVFP4 checkpoint, whose
            ignore list keeps *.linear_attn.* in BF16; sha256-verified extract).
  gdnbf16 : the BF16 originals (quality reference).
Every other file is a hardlink. The 4 shards that hold the FP8 GDN tensors are rewritten without them."""
import json, os, re, shutil, sys, hashlib
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/opt/llm/models/qwen38-flash-next-mtpfp4"
BF = os.path.expanduser("~jschmied/gdn4src/gdn_proj_bf16.safetensors")
OUTS = {"gdn4": "/opt/llm/models/qwen38-flash-next-mtpfp4-gdn4", "gdnbf16": "/opt/llm/models/qwen38-flash-next-mtpfp4-gdnbf16"}
PAT = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.")
MID = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
dev = "cuda"


def quant_nvfp4(w: torch.Tensor, g: float):
    N, K = w.shape
    grp = w.view(N, K // 16, 16)
    s = (grp.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = s.float() * g
    x = torch.where(sf[..., None] > 0, grp / sf[..., None].clamp(min=1e-30), torch.zeros_like(grp))
    code = torch.bucketize(x.abs().clamp(max=6.0), MID.to(w.device))
    nib = (code | ((x < 0).to(code.dtype) << 3)).to(torch.uint8).view(N, K)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous(), s.contiguous()


def dequant(q, s, g):
    N = q.shape[0]
    lo, hi = q & 0xF, q >> 4
    nib = torch.stack((lo, hi), -1).view(N, -1).long()
    mag = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=q.device)[nib & 7]
    v = mag * torch.where((nib >> 3) == 1, -1.0, 1.0)
    return (v.view(N, -1, 16) * s.float()[..., None] * g).view(N, -1)


idx_doc = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx_doc["weight_map"]
gdn_keys = sorted(k for k in wm if PAT.match(k))
affected = sorted({wm[k] for k in gdn_keys})
print("FP8 GDN tensors:", len(gdn_keys), "in shards:", affected, flush=True)
bf = safe_open(BF, "pt", device="cpu")
bf_by = {}
for k in bf.keys():
    m = re.search(r"layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$", k)
    bf_by[(int(m.group(1)), m.group(2))] = k
layers = sorted({int(PAT.match(k).group(1)) for k in gdn_keys})
assert len(layers) == 36 and len(bf_by) == 108, (len(layers), len(bf_by))

new = {"gdn4": {}, "gdnbf16": {}}
errs = []
for L in layers:
    ws = {p: bf.get_tensor(bf_by[(L, p)]).to(dev) for p in ("in_proj_qkv", "in_proj_z", "out_proj")}
    for p, w in ws.items():
        new["gdnbf16"][f"model.language_model.layers.{L}.linear_attn.{p}.weight"] = w.cpu().contiguous()
    g_qkvz = max(ws["in_proj_qkv"].float().abs().max().item(), ws["in_proj_z"].float().abs().max().item()) / 2688.0
    g_out = ws["out_proj"].float().abs().max().item() / 2688.0
    for p, w in ws.items():
        g = g_out if p == "out_proj" else g_qkvz
        q, s = quant_nvfp4(w.float(), g)
        e = ((dequant(q, s, g) - w.float()).norm() / w.float().norm()).item(); errs.append(e)
        base = f"model.language_model.layers.{L}.linear_attn.{p}"
        new["gdn4"][base + ".weight"] = q.cpu()
        new["gdn4"][base + ".weight_scale"] = s.cpu()
        new["gdn4"][base + ".weight_scale_2"] = torch.tensor([g], dtype=torch.float32)
print(f"NVFP4 rel weight error: mean {sum(errs)/len(errs):.4f} max {max(errs):.4f}", flush=True)

cfg = json.load(open(f"{SRC}/config.json"))
hq = json.load(open(f"{SRC}/hf_quant_config.json"))
for var, out in OUTS.items():
    if os.path.exists(out):
        sys.exit(f"{out} exists; remove it first")
    os.makedirs(out)
    skip = set(affected) | {"config.json", "hf_quant_config.json", "model.safetensors.index.json", "SHA256SUMS"}
    for f in os.listdir(SRC):
        if f in skip or os.path.isdir(f"{SRC}/{f}"):
            continue
        os.link(f"{SRC}/{f}", f"{out}/{f}")
    new_wm = {k: v for k, v in wm.items() if not PAT.match(k)}
    for sh in affected:
        keep = {}
        with safe_open(f"{SRC}/{sh}", "pt", device="cpu") as f:
            meta = f.metadata()
            for k in f.keys():
                if not PAT.match(k):
                    keep[k] = f.get_tensor(k)
        save_file(keep, f"{out}/{sh}", metadata=meta)
    gname = f"model-gdn-{var}.safetensors"
    save_file(new[var], f"{out}/{gname}", metadata={"format": "pt"})
    for k in new[var]:
        new_wm[k] = gname
    json.dump({"metadata": idx_doc.get("metadata", {}), "weight_map": new_wm}, open(f"{out}/model.safetensors.index.json", "w"), indent=2)
    c = json.loads(json.dumps(cfg)); h = json.loads(json.dumps(hq))
    for ql in (c["quantization_config"]["quantized_layers"], h["quantization"]["quantized_layers"]):
        for L in layers:
            for p in ("in_proj_qkv", "in_proj_z", "out_proj"):
                key = f"model.language_model.layers.{L}.linear_attn.{p}"
                assert key in ql, key
                if var == "gdn4":
                    ql[key] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
                else:
                    del ql[key]
    json.dump(c, open(f"{out}/config.json", "w"), indent=2)
    json.dump(h, open(f"{out}/hf_quant_config.json", "w"), indent=2)
    # every indexed tensor resolvable, none twice
    seen = {}
    for sh in sorted(set(new_wm.values())):
        with safe_open(f"{out}/{sh}", "pt", device="cpu") as f:
            for k in f.keys():
                assert k not in seen, ("dup", k); seen[k] = sh
    missing = [k for k, v in new_wm.items() if seen.get(k) != v]
    assert not missing, missing[:5]
    stray = [k for k in seen if k not in new_wm]
    json.dump({"variant": var, "base": SRC, "bf16_source": BF,
               "bf16_source_provenance": "RadixArk Qwen3.8-Flash-Next-NVFP4 on PBS /mnt/bulk/gb10/models/qwen38-flash-next-nvfp4 (linear_attn ignored -> BF16 originals); per-tensor sha256 in ~/gdn4src/SOURCE.json",
               "changed_tensors": len(new[var]), "rewritten_shards": affected, "gdn_shard": gname,
               "nvfp4_rel_err_mean": sum(errs) / len(errs) if var == "gdn4" else None},
              open(f"{out}/GDN_VARIANT.json", "w"), indent=1)
    print(var, "OK:", len(seen), "tensors,", len(stray), "stray (not indexed)", flush=True)
print("== BUILD DONE ==")
