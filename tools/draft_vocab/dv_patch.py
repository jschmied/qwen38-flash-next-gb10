# Reduced-vocabulary MTP drafting (FR-Spec style) for the FP8_PB_WO lm_head of the Flash-Next MTP module. Usage: dv_patch.py apply|revert <venv>
# Env FN_DRAFT_VOCAB=<file of token ids, one per line> -> the drafter's argmax runs over an exact BF16 dequant of those head rows.
import sys, shutil, os
mode, venv = sys.argv[1], sys.argv[2]; sp = f"{venv}/lib/python3.12/site-packages/vllm"
MTP = f"{sp}/models/qwen4_exp/nvidia/mtp.py"; PROP = f"{sp}/v1/spec_decode/llm_base_proposer.py"; FILES = [MTP, PROP]
def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old)); return s.replace(old, new)
HELPER = '''
import os as _fn_os
from vllm.logger import init_logger as _fn_init_logger
_fn_logger = _fn_init_logger(__name__)


def _fn_attach_draft_vocab(model) -> None:
    """Slice the drafter's FP8_PB_WO lm_head to the ids in FN_DRAFT_VOCAB (one per line) as an exact BF16 dequant.
    The full head is untouched; only get_top_tokens uses the slice."""
    path = _fn_os.environ.get("FN_DRAFT_VOCAB", "").strip()
    if not path:
        return
    head = model.lm_head
    w = getattr(head, "weight", None); s = getattr(head, "weight_scale", None)
    if w is None or s is None or w.dim() != 2 or s.dim() != 2 or w.dtype != torch.float8_e4m3fn:
        _fn_logger.warning("FNDV: lm_head is not a 2-D FP8 block-scaled head; skipping.")
        return
    if getattr(head, "tp_size", 1) != 1:
        _fn_logger.warning("FNDV: TP>1 unsupported; skipping."); return
    ids = sorted({int(x) for x in open(path).read().split() if x.strip()})
    N, K = w.shape; ids = [i for i in ids if 0 <= i < N]
    if not ids or len(ids) >= N:
        _fn_logger.warning("FNDV: %d usable ids against %d rows; skipping.", len(ids), N); return
    bs = getattr(head, "weight_block_size", [128, 128]); bn, bk = int(bs[0]), int(bs[1])
    assert K % bk == 0 and s.shape[0] == (N + bn - 1) // bn and s.shape[1] == K // bk, (bs, tuple(s.shape), (N, K))
    index = torch.tensor(ids, dtype=torch.long, device=w.device)
    with torch.no_grad():
        rows = w.index_select(0, index).to(torch.float32)
        srows = s.index_select(0, index // bn).to(torch.float32)
        deq = (rows * srows.repeat_interleave(bk, dim=1)).to(torch.bfloat16).contiguous()
    model.register_buffer("_fn_draft_weight", deq, persistent=False)
    model.register_buffer("_fn_draft_ids", index.to(torch.int64), persistent=False)
    mib = lambda t: t.numel() * t.element_size() / 2**20
    _fn_logger.info("FNDV draft vocab: %d of %d rows (%.1f%%); draft head %.0f -> %.0f MiB per draft step (BF16 slice of the FP8 head)",
                    len(ids), N, 100.0 * len(ids) / N, mib(w) + mib(s), mib(deq))

'''
GTT = '''
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft ids over the FN_DRAFT_VOCAB slice when attached (a dropped row can only cost a rejected draft,
        never a wrong output), else the full head via the local-argmax path. Logit scale 1, no soft cap: argmax is exact."""
        w = getattr(self, "_fn_draft_weight", None)
        if w is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        logits = torch.nn.functional.linear(hidden_states.to(w.dtype), w)
        return self._fn_draft_ids[logits.argmax(dim=-1)]

    def _fn_attach_draft_vocab(self) -> None:
        _fn_attach_draft_vocab(self)

'''
CL = "    def compute_logits(\n        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n    ) -> torch.Tensor | None:\n        return self.logits_processor(self.lm_head, hidden_states)\n"
LI = '            logger.info(\n                "Using local argmax reduction for draft token generation "\n                "(communication: O(2*tp_size) vs O(vocab_size))."\n            )\n'
if mode == "apply":
    for p in FILES: assert not os.path.exists(p + ".orig-dv"), p
    for p in FILES: shutil.copy2(p, p + ".orig-dv")
    s = open(MTP).read(); assert "FNDV" not in s
    s = rep(s, "def _remap_ignored_layers(\n", HELPER + "\ndef _remap_ignored_layers(\n")
    s = rep(s, CL, CL + GTT); open(MTP, "w").write(s)
    s = open(PROP).read(); assert "_fn_attach_draft_vocab" not in s
    s = rep(s, LI, LI + '            _fn_attach = getattr(self.model, "_fn_attach_draft_vocab", None)\n            if _fn_attach is not None:\n                _fn_attach()\n')
    open(PROP, "w").write(s); print("applied")
else:
    for p in FILES:
        b = p + ".orig-dv"; assert os.path.exists(b), b; shutil.copy2(b, p); os.remove(b)
    print("reverted")
