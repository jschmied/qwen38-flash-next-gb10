"""Local checks of four review claims against TensorFold PR #300 (run from its tests/ dir); each asserts the behaviour
the reviewer expects, so a failure confirms the claim."""

from __future__ import annotations

import pytest

from tensorfold.cuda.session_disk import DiskTier
from tensorfold.cuda.sessions import HostTier, TieredCache

from cuda_lane_fakes import Codec as LaneCodec  # noqa: F401  (imported for parity with the PR's tests)
from test_cuda_lanes import Pair, disk_tiers, prompts, serial, stream


class Codec:
    def to_host(self, state):
        import numpy as np
        return {"x": np.array(state, dtype=np.int64)}

    def from_host(self, arrays):
        return [int(v) for v in arrays["x"]]


def test_claim1_host_overflow_cascades_to_disk(tmp_path):
    """An entry the host tier displaces should reach the disk tier, not vanish."""

    entry = 8 * 4                                           # one int64 array of 4 values
    tiers = [HostTier(2 * entry), DiskTier(tmp_path, {"engine": "x"}, limit=1 << 20)]
    c = TieredCache(0, codec=Codec(), tiers=tiers)          # keep 0: every entry goes straight to the tiers
    for i, ids in enumerate(([1, 1], [2, 2], [3, 3])):
        c.add(ids, [i, i, i, i], None)
    assert len(tiers[0].keys()) == 2                        # the host holds two; [1, 1] was displaced
    assert c.find([1, 1, 9]) is not None, "the displaced host entry is nowhere: never cascaded to disk"


def test_claim5_same_length_unrelated_entry_stays_evictable():
    """Pinning the resumed entry should not pin every other entry of the same length."""

    c = TieredCache(4)
    c.add([1, 2, 3], "a", None)
    c.add([4, 5, 6], "b", None)                             # unrelated, same length
    assert c.victim(keep=(3,)) is not None, "every 3-token entry is protected, not just the resumed one"


def test_claim2_asymmetric_disk_does_not_break_the_follower(tmp_path):
    """Rank 0 should not resume from a disk entry rank 1 lacks."""

    p = prompts(1)[0]
    Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path)).run([stream(p, 6), stream(prompts(2)[1], 6)])
    for f in (tmp_path / "states").rglob("rank1/*.tfs"):   # rank 1's disk lost its entries (one rank's NVMe)
        f.unlink()
    pair = Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path))
    longer = p + [8, 8, 8]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)]


def test_claim3_a_failed_admit_on_rank0_leaves_the_ranks_agreeing(monkeypatch):
    """If rank 0's own ADMIT fails after the broadcast, the follower should not keep the lane."""

    pair = Pair(lanes=1, keep=0)
    real = pair.decoder.local._admit
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("rank 0 fails after the broadcast")
        return real(*a, **k)

    monkeypatch.setattr(pair.decoder.local, "_admit", flaky)
    with pytest.raises(RuntimeError):
        pair.decoder.admit(stream(prompts(1)[0], 4))
    assert all(lane is None for lane in pair.follower.lanes), "the follower kept a lane rank 0 gave back"
