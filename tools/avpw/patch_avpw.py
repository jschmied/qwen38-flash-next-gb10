"""FNAVPW: vLLM adaptive verification on our MTP + GDN RecoverSSM stack under PIECEWISE (breakable) graphs, c=1 only.
FN_AV_PW=1 relaxes the FULL-graph gates (config check, cudagraph mode forcing, ALWAYS builder support), profiles the
PIECEWISE capture sizes, prices the draft curve from all samples, and lets the GDN/PLE RecoverSSM backends accept
device-trimmed verify rows. QSA is NOT fixed for ragged rows (uniform CPU decode length): run with max_num_seqs=1.
Usage: python patch_avpw.py [on|off]; root = VLLM_PKG (default: clone venv package dir)."""
import os, sys
R = os.environ.get("VLLM_PKG", "/opt/llm/runtime/vllm-venv-rssm/lib/python3.12/site-packages/vllm")
MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
E = '__import__("os").environ.get("FN_AV_PW", "") == "1"'
EDITS = [
 ("config/vllm.py",
  '''        if not self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise ValueError(
                "Adaptive verification requires full CUDA graphs.''',
  f'''        if not self.compilation_config.cudagraph_mode.has_full_cudagraphs() and not {E}:  # FNAVPW
            raise ValueError(
                "Adaptive verification requires full CUDA graphs.'''),
 ("v1/worker/gpu/model_runner.py",
  '''        if self.adaptive_verification is not None:
            self.compilation_config.cudagraph_mode = resolve_adaptive_cudagraph_mode(''',
  f'''        if self.adaptive_verification is not None and not {E}:  # FNAVPW
            self.compilation_config.cudagraph_mode = resolve_adaptive_cudagraph_mode('''),
 ("v1/worker/gpu/model_runner.py",
  '''                            for batch in self.adaptive_verification.batches_to_profile(
                                self.cudagraph_manager.captured_token_counts()
                            ):''',
  f'''                            for batch in self.adaptive_verification.batches_to_profile(
                                sorted(self.compilation_config.cudagraph_capture_sizes or [])  # FNAVPW
                                if {E}  # FNAVPW
                                else self.cudagraph_manager.captured_token_counts()
                            ):'''),
 ("v1/worker/gpu/spec_decode/adaptive_verification.py",
  '''            (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph
        )''',
  f'''            (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph or {E}  # FNAVPW
        )'''),
 ("v1/worker/gpu/spec_decode/adaptive_verification.py",
  '''            tail_sizes -= set(capture_sizes)
''',
  f'''            tail_sizes -= set(capture_sizes)
        if {E}:  # FNAVPW: c=1 decode budgets never exceed the capture sizes (K+1); larger dummy batches would
            tail_sizes = set()  # FNAVPW  exceed the per-request verify window of the RecoverSSM builders
'''),
 ("v1/worker/gpu/spec_decode/adaptive_verification.py",
  '''        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
''',
  f'''        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        if {E}:  # FNAVPW
            logger.warning("FNAVPW adaptive verification: draft curve %s; verify curve %s",  # FNAVPW
                           draft_curve, verify_curve)  # FNAVPW
'''),
 ("v1/worker/gpu/spec_decode/adaptive_verification.py",
  '''    if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS:''',
  f'''    if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS and not {E}:  # FNAVPW'''),
 ("v1/attention/backends/gdn_recoverssm.py",
  '''class GDNRecoverSSMAttentionBackend(GDNAttentionBackend):
''',
  f'''class GDNRecoverSSMAttentionBackend(GDNAttentionBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:  # FNAVPW
        return {E} or super().supports_device_cpu_query_lens_mismatch()  # FNAVPW

'''),
 ("v1/attention/backends/ple_recoverssm.py",
  '''class PleRecoverSSMAttentionBackend(PleShortConvAttentionBackend):
''',
  f'''class PleRecoverSSMAttentionBackend(PleShortConvAttentionBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:  # FNAVPW
        return {E} or super().supports_device_cpu_query_lens_mismatch()  # FNAVPW

'''),
]
files = {}
for rel, old, new in EDITS:
    p = os.path.join(R, rel)
    files.setdefault(p, open(p).read())
for rel, old, new in EDITS:
    p = os.path.join(R, rel); s = files[p]
    a, b = (old, new) if MODE == "on" else (new, old)
    if s.count(a) != 1:
        sys.exit(f"{MODE}: anchor count {s.count(a)} in {rel}: {a[:60]!r}")
    files[p] = s.replace(a, b)
for p, s in files.items():
    if MODE == "off" and "FNAVPW" in s: sys.exit(f"leftover marker in {p}")
    open(p, "w").write(s)
print("FNAVPW", MODE, len(files), "files")
