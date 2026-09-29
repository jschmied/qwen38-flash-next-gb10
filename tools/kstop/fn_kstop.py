# SPDX-License-Identifier: Apache-2.0
# FNKSTOP (2026-09-29, speed-of-light §5ah): confidence-stopped MTP drafting on the V2 model runner.
#   - update(): after each draft step, on the GPU, a request stays active while the drafter's top-1 probability is
#     >= FN_KSTOP_TAU; d = the number of kept drafts (the first draft is always kept). Graph-padding rows are masked by
#     the real request count (set_nreq, host side, outside the graphs).
#   - pred(): the device bool "any request still active". The drafter's FULL decode-step graphs are captured inside
#     a CUDA-graph IF node on it, so the GPU skips the remaining draft steps once every request has stopped.
#   - copy_d_async() / host_d(): d_max goes to the host once per step, and the scheduler receives [-1] * d_max drafts
#     per request, so the next verify runs exactly 1 + d_max rows (needs --no-async-scheduling and a FULL verify graph
#     per draft count, i.e. a dynamic-SD schedule; FN_KSTOP_SIZE=0 keeps K rows and only skips draft steps).
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("FN_KSTOP") == "1"
TAU = float(os.environ.get("FN_KSTOP_TAU", "0.75"))
SIZE = os.environ.get("FN_KSTOP_SIZE", "1") == "1"
_S: dict = {}


def _buf(max_reqs: int, device) -> dict:
    if not _S:
        _S["active"] = torch.zeros(max_reqs, dtype=torch.bool, device=device)
        _S["d"] = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        _S["any"] = torch.zeros((), dtype=torch.bool, device=device)
        _S["dmax"] = torch.zeros((), dtype=torch.int32, device=device)
        _S["nreq"] = torch.zeros((), dtype=torch.int64, device=device)
        _S["ar"] = torch.arange(max_reqs, device=device)
        _S["host"] = torch.zeros((1,), dtype=torch.int32, pin_memory=True)
        _S["event"] = torch.cuda.Event()
        _S["hist"] = {}
        _S["cycles"] = 0
        logger.info("FNKSTOP active: tau %.2f, verify sized to the draft count: %s", TAU, SIZE)
    return _S


def pred(max_reqs: int, device) -> torch.Tensor:
    logger.info("FNKSTOP IF-node draft decode graphs")
    return _buf(max_reqs, device)["any"]


def set_nreq(num_reqs: int, max_reqs: int, device) -> None:
    _buf(max_reqs, device)["nreq"].fill_(num_reqs)


def update(logits: torch.Tensor, draft_step: torch.Tensor) -> None:
    """Graph-safe: tensor ops on persistent buffers only. logits: [num_reqs(+graph padding), vocab]."""
    if not _S:
        return
    n = logits.shape[0]
    conf = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
    valid = _S["ar"][:n] < _S["nreq"]
    is0 = draft_step == 0
    act = _S["active"][:n]
    d = _S["d"][:n]
    new_act = torch.where(is0, valid, act & (conf >= TAU) & valid)
    new_d = torch.where(is0, valid.to(torch.int32),
                        torch.where(new_act, (draft_step + 1).to(torch.int32), d))
    act.copy_(new_act)
    d.copy_(new_d)
    _S["any"].copy_(new_act.any())
    _S["dmax"].copy_(new_d.max())


def copy_d_async() -> None:
    if not _S:
        return
    _S["host"].copy_(_S["dmax"].view(1), non_blocking=True)
    _S["event"].record()


def host_d(default: int) -> int:
    """The one host sync per step: the next verify's draft count."""
    if not _S:
        return default
    _S["event"].synchronize()
    v = int(_S["host"][0])
    v = default if v <= 0 else min(v, default)
    h = _S["hist"]
    h[v] = h.get(v, 0) + 1
    _S["cycles"] += 1
    if _S["cycles"] == 1:
        logger.info("FNKSTOP verify sized to the draft count (first cycle: %d of %d)", v, default)
    if _S["cycles"] % 2000 == 0:
        logger.info("FNKSTOP draft-count histogram after %d cycles: %s", _S["cycles"], dict(sorted(h.items())))
    return v
