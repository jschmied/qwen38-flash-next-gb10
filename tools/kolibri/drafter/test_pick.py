"""train_block.pick: a balanced mix stops at the first empty class; with drain the rest keep their shares."""

from train_block import pick


def items(swe, chat, de):
    return [("r", i, c) for c, n in (("swe", swe), ("chat", chat), ("de", de)) for i in range(n)]


def test_balanced_mix_stops_when_one_class_runs_out():
    it, done, mix = items(10, 10, 2), {}, {"swe": 0.3, "chat": 0.4, "de": 0.3}
    taken = []
    while (got := pick(it, mix, done)) is not None:
        taken.append(got[2])
    assert taken.count("de") == 2 and len(it) > 0       # stopped with swe/chat left over


def test_drain_keeps_the_other_classes_in_their_ratio():
    it, done, mix = items(30, 40, 2), {}, {"swe": 0.3, "chat": 0.4, "de": 0.3}
    taken = []
    while (got := pick(it, mix, done, drain=True)) is not None:
        taken.append(got[2])
    assert len(it) == 0 and taken.count("de") == 2       # everything taken
    tail = taken[20:]                                     # after German ran out: swe : chat ~ 3 : 4
    assert abs(tail.count("swe") / max(1, tail.count("chat")) - 0.75) < 0.2


if __name__ == "__main__":
    test_balanced_mix_stops_when_one_class_runs_out()
    test_drain_keeps_the_other_classes_in_their_ratio()
    print("ok")
