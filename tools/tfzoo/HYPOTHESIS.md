# Does TensorFold handle every quant combination in our zoo? (2026-10-01, user: "we have a zoo of different quants for
# some components. we could try if all combinations are handled properly"), written before the run

Every Flash-Next checkpoint variant on disk, one start each on stock 0.6.0 (`--context 32768 --parallel 1`): does it
refuse with a clear message, fail at load, or load; if it loads, are greedy replies sane (3 fixed prompts) and is
the MTP acceptance plausible (`/health` drafted/accepted)? Then PR #176's branch on the variants it should change.
Expected per variant (stock):
- `fp8head` (block-FP8 dense + head, NVFP4 experts, bf16 MTP): loads, sane, acceptance 55–75 %.
- `mtpfp4` (+ NVFP4 MTP experts): loads, sane, acceptance 55–75 % (served before).
- `mtpfp4-gdnbf16` (bf16 DeltaNet): loads, sane.
- `mtpfp4-plebf16` (bf16 n-gram table): loads, sane.
- `exl3`: loads, sane (served before).
- `mtpfp8` (per-tensor FP8 MTP experts): **refused at the check** (`FP8` is not an accepted algo).
- `mtpfp4d` (NVFP4 MTP dense layers + fc): the check accepts it; the MTP dense weights reach `weight_bf16`'s cast as
  packed bytes → **load error on shapes or garbage drafts** (acceptance near 0, replies still correct: the target
  verifies every draft).
- `mtpfp4-gdn4` (W4A16 NVFP4 DeltaNet projections): the check accepts it; `dense()` returns packed bytes as a bf16
  linear → **load error on shapes or garbage replies**.
- Branch #176: `mtpfp4d` and `mtpfp4-gdn4` refused at the check, naming a layer; every other variant unchanged.
Out of range (e.g. a variant that loads and decodes garbage without an error) → that is a finding for its own issue.
