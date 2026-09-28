"""Build qwen38-flash-next-mtpfp4-ablit: our prod checkpoint with dealignai's abliterated o_proj weights in our format.
12 main-model self_attn.o_proj: FP8 E4M3 blockwise 128x128 from their BF16 with the recipe that reproduces our
checkpoint bit-exactly (scale = amax * (1/448), weight_scale_inv = scale); mtp.layers.0.self_attn.o_proj: their BF16 copied.
The 3 shards are copied (not hardlinked) and only these tensors' byte ranges are overwritten. argv: <scratch dir>"""
import json, os, shutil, struct, sys, torch
from safetensors import safe_open
SRC = "/opt/llm/models/qwen38-flash-next-mtpfp4"; DST = "/opt/llm/models/qwen38-flash-next-mtpfp4-ablit"; S = sys.argv[1]
SHARDS = ["model-bf16-00010.safetensors", "model-bf16-00011.safetensors", "model-bf16-00012.safetensors"]
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
def quant(w):
    R, C = w.shape; b = w.float().reshape(R // 128, 128, C // 128, 128).permute(0, 2, 1, 3)
    scale = b.abs().amax(dim=(2, 3)) * (1.0 / 448.0)
    q = (b / scale[:, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.permute(0, 2, 1, 3).reshape(R, C).contiguous(), scale.contiguous()
os.makedirs(DST, exist_ok=False)
for f in os.listdir(SRC):
    if f in SHARDS or f == "SHA256SUMS": continue
    os.link(f"{SRC}/{f}", f"{DST}/{f}")
for f in SHARDS: shutil.copyfile(f"{SRC}/{f}", f"{DST}/{f}")
new = {}
with safe_open(f"{S}/ablit_oproj.safetensors", "pt") as ab:
    for n in ab.keys():
        w = ab.get_tensor(n)
        if n.startswith("mtp."): new[n] = w.contiguous()
        else:
            q, s = quant(w); new[n] = q; new[n + "_scale_inv"] = s
hdrs = {}
for f in SHARDS:
    fh = open(f"{DST}/{f}", "r+b"); k = struct.unpack("<Q", fh.read(8))[0]; hdrs[f] = (fh, 8 + k, json.loads(fh.read(k)))
done = 0
for n, t in new.items():
    fh, base, h = hdrs[wm[n]]; v = h[n]
    exp = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}[v["dtype"]]
    assert t.dtype == exp and list(t.shape) == v["shape"], (n, t.dtype, t.shape, v)
    a, b = v["data_offsets"]; data = t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.float8_e4m3fn else t.contiguous().view(torch.uint8).numpy().tobytes()
    assert len(data) == b - a, (n, len(data), b - a)
    fh.seek(base + a); fh.write(data); done += 1
for fh, _, _ in hdrs.values(): fh.close()
# read back
ok = 0
for n, t in new.items():
    with safe_open(f"{DST}/{wm[n]}", "pt") as f: r = f.get_tensor(n)
    ok += int(torch.equal(r.view(torch.uint8), t.view(torch.uint8)) if t.dtype == torch.float8_e4m3fn else torch.equal(r, t))
print(json.dumps({"written": done, "readback_equal": ok, "tensors": len(new)}))
