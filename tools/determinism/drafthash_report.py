# Bisect bug B inside the drafter. argv: <server log> <mtpdh.txt> <ARM>
# Groups DRAFTHASH lines into agent-loop turns (a prompt's prefill chunks + the decode calls that
# follow), assigns pass/turn (3 passes x 8 turns), and for every turn compares passes pairwise:
# first differing module per call, plus whether the drafter INPUT already differed.
import re, sys, collections
log, outf, arm = sys.argv[1:4]
# state per (pass, turn) from the results file
state = {}
for ln in open(outf, errors="replace"):
    m = re.search(rf"{arm} pass(\d) turn (\d):.*acc= *([\d.]+)%", ln)
    if m: state[(int(m[1]), int(m[2]))] = "F" if float(m[3]) >= 40 else "s"
# parse hashes: dh_turn -> call -> ordered list of (name, hash) ; INPUT carries ntok
calls = collections.defaultdict(lambda: collections.defaultdict(list)); ntok = {}
for ln in open(log, errors="replace"):
    m = re.search(r"DRAFTHASH turn=(\d+) call=(\d+) (\S+) (.*)", ln)
    if not m: continue
    t, c, name, rest = int(m[1]), int(m[2]), m[3], m[4]
    if name == "INPUT":
        n = re.search(r"ntok=(\d+)", rest); ntok[(t, c)] = int(n[1])
        h = re.search(r"hid_row0=(\w+)", rest); ids = re.search(r"ids=(\[[^\]]*\])", rest)
        calls[t][c].append(("INPUT.hidden", h[1])); calls[t][c].append(("INPUT.ids", ids[1] if ids else "?"))
    else:
        h = re.search(r"row0=(\w+)", rest); calls[t][c].append((name, h[1] if h else "?"))
# warmup: dh-turns before the first real 4096-token prefill at pos 0 (warmup ids are [0,0,0] / [1,2,3])
ids = {}
for ln in open(log, errors="replace"):
    m = re.search(r"DRAFTHASH turn=(\d+) call=1 INPUT ntok=(\d+) ids=(\[[^\]]*\]) pos=\[(\d+)\]", ln)
    if m: ids[int(m[1])] = (int(m[2]), m[3], int(m[4]))
first_real = next(t for t in sorted(ids) if ids[t][0] == 4096 and ids[t][2] == 0 and ids[t][1] not in ("[0, 0, 0]", "[1, 2, 3]"))
# an agent turn = prefill chunk(s) ... closed by the short post-prompt call (ntok <= 64)
groups, cur = [], []
for t in sorted(calls):
    if t < first_real: continue
    cur.append(t)
    if ids.get(t, (999,))[0] <= 64:
        groups.append(cur); cur = []
if cur: groups.append(cur)
print(f"{arm}: {len(groups)} agent turns from {len(calls) - first_real + 1} draft-prefill events (warmup skipped: {first_real - 1}); expected 24")
def sig(g):  # whole group: (dh offset, call) -> module list; keyed so passes align even if chunking differs
    out = {}
    for k, t in enumerate(g):
        for c in sorted(calls[t]): out[(k - len(g), c)] = calls[t][c]   # offset from the END of the group
    return out
for i, g in enumerate(groups[:24]):
    p, tn = i // 8 + 1, i % 8 + 1
    groups[i] = (p, tn, sig(g))
byturn = collections.defaultdict(dict)
for p, tn, s in groups[:24]: byturn[tn][p] = s
for tn in sorted(byturn):
    ps = byturn[tn]
    for a in sorted(ps):
        for b in sorted(ps):
            if b <= a: continue
            sa, sb = state.get((a, tn), "?"), state.get((b, tn), "?")
            firsts = []
            for c in sorted(set(ps[a]) | set(ps[b])):
                lb = dict(ps[b].get(c, []))
                diff = next((n for n, h in ps[a].get(c, []) if lb.get(n) != h), None)
                firsts.append(f"{c[0]}/c{c[1]}:{'=' if diff is None else diff}")
            print(f"turn {tn} pass{a}({sa}) vs pass{b}({sb}): " + "  ".join(firsts))
