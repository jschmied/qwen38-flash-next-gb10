ABLIT serve test (2026-09-28 ~19:30, user: "there is dealignai/...ABLITERATED-NVFP4 maybe requant some components to
same as our current model"). Checkpoint qwen38-flash-next-mtpfp4-ablit (see its PROVENANCE.md: 25 tensors differ from
prod's, the FP8 recipe reproduces ours bit-exactly). One start on prod's venv + launcher + drop-ins, FN_MODEL swapped,
KV 4 GiB, probe nvprobe.
H: loads, every prod path line present (HC fusion, FULL capture, prefilter, RecoverSSM, FNNVFP4...); speed equal to prod
within noise (code c=1 14.6..14.9 ms/tok, prose 23.4..23.9, TTFT 8k 2.70..2.75 s); acceptance within +-3 % of prod
(4.365 code / 2.856 prose; the MTP o_proj changed too, per the card "draft head agrees with the model"); greedy hashes
may differ from prod's (13 weights differ). Out of range: load failure, a missing path line, or acceptance down > 5 %.
