"""Speed-of-light byte budget for one MTP n=3 verify cycle at c=1 (Flash-Next, prod checkpoint mtpfp4 + NVFP4 draft head).
Byte sizes from the safetensors headers (tensors.json); state/KV sizes from config.json. Floor = bytes / BW."""
import json, re
rows = json.load(open("tensors.json"))
MiB = 2**20
def grp(pred): return sum(nb for k, dt, sh, nb in rows if pred(k)) / MiB
L_ = "model.language_model.layers."
dense_fp8 = grp(lambda k: k.startswith(L_) and ".experts." not in k and "ngram_embedding" not in k and
                any(s in k for s in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                                     "self_attn.q_proj", "self_attn.o_proj", "self_attn.k_proj", "self_attn.v_proj")))
dense_all = grp(lambda k: k.startswith(L_) and ".experts." not in k and "ngram_embedding" not in k)
top_mixer = grp(lambda k: k.startswith("model.language_model.hyper_connection_mixer"))
dense_other = dense_all - dense_fp8 + top_mixer          # BF16 leftovers + norms + scales
hc = grp(lambda k: k.startswith(L_) and "hyper_connection" in k)
shared = grp(lambda k: k.startswith(L_) and "shared_expert" in k)
router = grp(lambda k: k.startswith(L_) and k.endswith("mlp.gate.weight"))
lm_head = grp(lambda k: k.startswith("lm_head"))
exp_layer = grp(lambda k: k.startswith(L_ + "0.mlp.experts."))
per_expert = exp_layer / 512
mtp_dense = grp(lambda k: k.startswith("mtp.") and ".experts." not in k)
mtp_exp = grp(lambda k: k.startswith("mtp.") and ".experts." in k) / 512
draft_head = 45.0                                         # NVFP4 32k slice (finding 234)
gdn_state = 36 * 48 * 128 * 128 * 4 / MiB                 # fp32 recurrent state, all GDN layers
print(f"target dense FP8 {dense_fp8:.0f} MiB | other dense (BF16 leftovers etc.) {dense_other:.0f} MiB "
      f"[hc {hc:.0f}, shared {shared:.0f}, router {router:.0f}] | lm_head {lm_head:.0f} | per expert {per_expert:.3f} MiB")
print(f"MTP dense {mtp_dense:.0f} MiB (BF16) | MTP per expert {mtp_exp:.3f} MiB | draft head {draft_head} MiB | GDN state {gdn_state:.0f} MiB")
BW = 220e9
def cycle(E4, ctx, gdn_writes):
    sel = min(ctx, 2048)
    qsa = 12 * sel * 2 * 256 * 2 * 2 / MiB + 12 * (ctx / 4) * 128 * 2 / MiB        # K+V of selected + indexer keys
    tgt = dense_fp8 + dense_other + lm_head + E4 * 48 * per_expert + gdn_state * (1 + gdn_writes) + qsa
    mtp_qsa = sel * 2 * 256 * 2 * 2 / MiB + (ctx / 4) * 128 * 2 / MiB
    drf = 3 * (mtp_dense + draft_head + mtp_qsa) + (E4 + 2 * 10) * mtp_exp
    return tgt, drf
measured_cycle_ms = 23.36 * 2.535                         # finding 234, NVFP4 head arm, c=1 essays
print(f"\nmeasured c=1: 23.36 ms/tok x 2.535 tok/cycle = {measured_cycle_ms:.1f} ms per verify cycle\n")
print(f"{'E_distinct(4 tok)':>17} {'ctx':>6} {'GDN writes':>10} | {'target MiB':>10} {'draft MiB':>9} | {'floor ms/cycle':>14} {'floor ms/tok':>12} | measured/floor")
for E4 in (10, 20, 30, 39.3):
    for ctx in (1024, 8192, 32768):
        for gw in (1, 4):
            t, d = cycle(E4, ctx, gw)
            ms = (t + d) * MiB / BW * 1e3
            print(f"{E4:17} {ctx:6d} {gw:10d} | {t:10.0f} {d:9.0f} | {ms:14.1f} {ms/2.535:12.2f} | {measured_cycle_ms/ms:5.2f}x")
