PRODVAL3 (2026-09-28 ~15:15, user: "yes, both goes to prod"). Validation of the prod venv + launcher + drop-ins after
installing HC fusion and F4 (drop-in 55: FN_HCFUSE=1, FN_CG_MODE=FULL_AND_PIECEWISE). One start, default KV (as
prodval2 for comparability). Void rules: every prodval2 path line plus "FNHCFUSE fused hyper-connection kernels ran"
and "Capturing CUDA graphs (FULL)".
H: all path lines present; code/prose c=1 hashes identical to prodval2's (d102a738 / 38c70791: short prompts stay
below 128 tokens per batch... and F4 is output-identical at c=1); TTFT 8k 2.886 -> 2.65..2.80 s, 30k 10.39 ->
9.5..10.0 s; code c=1 14.74 -> 14.4..14.7 ms/tok. One start, so a validation, not an A/B.
Out of range: any missing path line (VOID), a hash change on the short-prompt probes, or TTFT not lower.
