# Compatibility — which base checkpoints this head fits

This FP8 head was quantized from **one specific BF16 `lm_head` tensor**. Dropping it onto a
checkpoint carrying a different one produces **wrong logits, not an error** — nothing will crash,
nothing will warn, and the model will keep generating fluent text. So check first.

## The reference

Your base checkpoint's `lm_head.weight` must be this exact tensor:

| | |
| --- | --- |
| dtype / shape | `BF16 (248320, 2560)` |
| **sha256 of the tensor's bytes** | `40bddd25d0d94a128ab08280faad39cfc3ee3064252761269115f544722607c9` |

Verified 2026-09-10 to be **bit-identical** in:

- [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the BF16 parent
- [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) —
  ships Qwen's head unchanged

That is the whole basis for the claim. It is **two checkpoints**, not a survey of the field, and it
is a statement about 2026-09-10 — nothing stops a future build from shipping a different head. Run
the check rather than trusting the list.

## The check

Hashes one tensor. Reads ~1.18 GiB, not the whole checkpoint.

```python
#!/usr/bin/env python3
"""Does this checkpoint carry the lm_head the FP8 head was built from?"""
import hashlib, json, os, sys

REF = "40bddd25d0d94a128ab08280faad39cfc3ee3064252761269115f544722607c9"
d = sys.argv[1]                                    # path to the base checkpoint

wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
name = next(k for k in wm if k.endswith("lm_head.weight"))
shard = os.path.join(d, wm[name])

with open(shard, "rb") as f:
    n = int.from_bytes(f.read(8), "little")
    hdr = json.loads(f.read(n))
    a, b = hdr[name]["data_offsets"]
    f.seek(8 + n + a)
    h, left = hashlib.sha256(), b - a
    while left:
        blk = f.read(min(1 << 24, left))
        h.update(blk); left -= len(blk)

got = h.hexdigest()
print(f"{name}: {hdr[name]['dtype']} {tuple(hdr[name]['shape'])}")
print(f"  found    {got}")
print(f"  expected {REF}")
print("  COMPATIBLE" if got == REF else "  *** MISMATCH — do not use this head with this base ***")
sys.exit(0 if got == REF else 1)
```

## If it mismatches

Do not use the head. Two things it could mean, and neither is safe to guess at:

- the base ships a different BF16 head (a fine-tune, a vocabulary change, a re-export), or
- the base already quantized its head, in which case you do not need this one.

Either way the fix is to re-quantize from *your* base rather than to reuse this tensor. The method is
written up in
[quantizing-lm-head.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md),
including the trap that decides it: ModelOpt's `weight_scale_inv` holds the **scale, not the
reciprocal**, and getting that backwards costs 565,100,324 % relative error instead of 2.2489 %.

## Also check the merge itself

Separate failure, same silence: if any shard your index still references contains an `lm_head`
tensor, it will **overwrite** this one — `safetensors_weights_iterator` walks every tensor in every
referenced file, and later files win by `_natural_sort_key`. In the RadixArk build `lm_head` shares
a shard with 169 other tensors, so that shard has to be repacked without it. See the README's merge
section.
