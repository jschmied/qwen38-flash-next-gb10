#!/usr/bin/env python3
"""Compare logit signatures ACROSS arms to separate 'unstable' from 'wrong'.

Every arm sends the SAME fixed prompt to the logit probe, so signatures are comparable between
arms whenever the serving config matches. That makes one comparison decisive:

  cache-off is stable at X, cache-on scatters around X   -> reuse is NONDETERMINISTIC
  cache-off is stable at X, cache-on sits at Y != X      -> reuse is systematically WRONG

The second is the case that costs quality: it means every cached request returns a different
answer than the same request computed from scratch, consistently, and no amount of repeating a
cached run would reveal it -- you can only see it by comparing against the uncached baseline.

Only arms with a matching config are compared. NOPFX_* (prefix cache off) and M_align_* /
M_all_* (prefix cache on) are all `mtp 5 / batch 4096` with no experiment hooks, so they form a
clean set. Earlier arms (P_ctl etc.) carried VLLM_QSA_TORCH_TOPK and are NOT comparable -- they
are listed separately and must not be mixed into the verdict.
"""
import re, glob, os, sys, collections

S = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
COMPARABLE = ("NOPFX_", "M_all_", "M_align_")

sigs = collections.defaultdict(list)
lps = collections.defaultdict(list)
for f in glob.glob(os.path.join(S, "*.txt")):
    for line in open(f, errors="replace"):
        m = re.search(r"LOGIT (\S+) req(\d+): top='(.*?)' lp=(-?[\d.]+) sig=([0-9a-f]+)", line)
        if m:
            sigs[m.group(1)].append(m.group(5))
            lps[m.group(1)].append(float(m.group(4)))

def show(title, arms):
    print(f"\n  === {title} ===")
    if not arms:
        print("    (none yet)")
        return
    for a in sorted(arms):
        s, l = sigs[a], lps[a]
        uniq = sorted(set(s))
        tag = "STABLE" if len(uniq) == 1 else f"{len(uniq)} distinct"
        print(f"    {a:<14} {tag:<12} lp={min(l):.6f}..{max(l):.6f}  sig={','.join(x[:8] for x in uniq[:3])}")

comp = [a for a in sigs if a.startswith(COMPARABLE)]
other = [a for a in sigs if not a.startswith(COMPARABLE)]
show("comparable set (mtp 5, batch 4096, no experiment hooks)", comp)
show("NOT comparable (different config / experiment hooks) -- context only", other)

off = {s for a in comp if a.startswith("NOPFX_") for s in sigs[a]}
on = {s for a in comp if a.startswith(("M_align_", "M_all_")) for s in sigs[a]}
print("\n  === verdict ===")
if not off or not on:
    print("    need both cache-off (NOPFX_*) and cache-on (M_*) arms before this can be decided")
else:
    print(f"    cache OFF: {len(off)} distinct signature(s) {sorted(x[:8] for x in off)}")
    print(f"    cache ON : {len(on)} distinct signature(s) {sorted(x[:8] for x in on)[:6]}")
    if len(off) == 1 and off <= on and len(on) > 1:
        print("    -> cache-on SCATTERS AROUND the uncached answer: reuse is NONDETERMINISTIC.")
    elif len(off) == 1 and not (off & on):
        print("    -> cache-on NEVER produces the uncached answer: reuse is systematically WRONG.")
        print("       This is a QUALITY defect, not just a reproducibility one: every cached")
        print("       request differs from the same request computed from scratch.")
    elif off == on:
        print("    -> identical: the prefix cache changes nothing here; look elsewhere.")
    else:
        print("    -> mixed; report the sets above without a one-line verdict.")
