# Fetching 12.45 GiB instead of 123.4

Switching from `RadixArk/Qwen3.8-Flash-Next-NVFP4` to
`lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` looks like a 123.4 GiB download. It is **12.45 GiB**,
because the second is a fork of the first with four shards rewritten.

## How to tell, without downloading anything

HuggingFace publishes `lfs.sha256` for every LFS file in the same API response that carries sizes.
Two repos can therefore be diffed exactly, for free:

```bash
for r in RadixArk/Qwen3.8-Flash-Next-NVFP4 lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8; do
  curl -s "https://huggingface.co/api/models/$r?blobs=true" \
    | jq -r '.siblings[]|select(.lfs)|"\(.lfs.sha256)  \(.rfilename)"' | sort > /tmp/$(basename $r).sums
done
join -j2 /tmp/*.sums | awk '$2!=$3{print $1}'     # files that actually differ
```

Result here:

```
206 shards in common
  byte-IDENTICAL : 202   (111.0 GiB — already on disk)
  differing      :   4   ( 12.4 GiB to fetch)
```

The four are `model-bf16-0000{1,10,11,12}.safetensors` — exactly the BF16 body shards holding the
dense weights. Even the 192 `layer-*-experts-*.complete.json` audit files and
`qualification-notes.md` are shared, which is how you can tell one repo was forked from the other
rather than converted independently from the base model.

## Building it

Hardlink what you have, fetch what you don't. Hardlinks cost **zero** additional disk on the same
filesystem — free space was unchanged at 99 GiB after linking 410 files.

```bash
# link the identical files, then aria2 the rest
python3 build_slice.py          # see scripts/
aria2c -i slice.aria2 -d $DST --max-overall-download-limit=6M -x1 -s1 -c
```

Metadata that must come from the new repo, because BF16 tensors become FP8 plus scale tensors and
the names change: `model.safetensors.index.json` (35 MB), `config.json`, `hf_quant_config.json`.

## Then verify by checksum, not size

All 206, against the **new** repo's published hashes — including the 202 you did not download,
because a hardlink is only as good as the file it points at:

```bash
sha256sum -c SHA256SUMS
```

## Before you serve it: check the runtime dispatches the format

This checkpoint declares `quant_algo: "FP8_PB_WO"` (blockwise 128x128 weight-only FP8) for its
attention and GDN projections. If your runtime does not dispatch that algorithm it will not
error — it will load the packed FP8 bytes into BF16 parameters and serve fluent garbage. Both
stock SGLang and vLLM snapshots older than the fix do exactly this.

See [failure-modes.md#a2b](failure-modes.md) for the ten-second offline check and the one-line
fix. Do this **before** the download, not after.

## Why this generalises

Quantization repos are usually forks: someone takes an existing conversion and re-does one axis.
The expensive, unchanged bulk — experts, embedding tables — is often byte-identical. Diffing
`lfs.sha256` between two repos costs one API call each and routinely turns a 100+ GiB download
into a handful of gigabytes. It also gives you, for free, an exact statement of what the second
author actually changed — which is better documentation than most model cards carry.
