#!/usr/bin/env python3
"""Find the first module whose output differs across identical forward passes.

Reads LAYERHASH lines emitted by layerhash_patch.py (from the server log or an arm results file).

THE METHOD THAT MAKES THIS WORK: passes are grouped by an INPUT FINGERPRINT before anything is
compared. A server log interleaves prefill and decode passes with different inputs; comparing them
directly says "everything differs, including layer 0", which points at embeddings and is wrong.
The first hooked module's hash identifies the input, so passes sharing it received the same input
and can legitimately be compared. This distinction turned a garbage answer into a clean one on
2026-09-01 -- do not remove it.

Also drops "dummy" passes: startup profiling runs feed zeros, so every module hashes alike. A pass
whose modules produce <=2 distinct hashes is profiling, not a real request.

RACE flags (from layerhash_patch.py hashing each tensor twice with a sync between) are reported
separately: a module that hashes differently twice in a row is being written asynchronously, and
its "identical across passes" result cannot be trusted.

Usage: layerhash_report.py <file> [fingerprint-module-substring]
       default fingerprint is the module ending in "layers.0"
"""
import re, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "-"
fp_hint = sys.argv[2] if len(sys.argv) > 2 else "layers.0"
text = (sys.stdin if path == "-" else open(path, errors="replace")).read()

rec = re.findall(r"LAYERHASH pass=(\d+) (\S+) ([0-9a-f]+)(\s+RACE h2=[0-9a-f]+)?", text)
if not rec:
    sys.exit("no LAYERHASH lines found")

by = collections.defaultdict(dict)
races = collections.Counter()
for p, mod, h, race in rec:
    by[int(p)][mod] = h
    if race:
        races[mod] += 1

passes = sorted(by)
real = [p for p in passes if len(set(by[p].values())) > 2]
print(f"  {len(rec)} hash lines, {len(passes)} passes, {len(real)} real "
      f"({len(passes)-len(real)} dummy/profiling)")

if races:
    print("\n  ASYNC RACE DETECTED -- these modules hashed differently twice with a sync between:")
    for m, n in races.most_common():
        print(f"    {m}  ({n} occurrences)")
    print("  Any 'identical' verdict for these modules is unreliable: the value changes after"
          "\n  the hook fires, so the consumer may read something else.")

def order(m):
    x = re.search(r"layers\.(\d+)(?:\.|$)", m)
    return (int(x.group(1)) if x else 9999, m)

fp_mod = next((m for m in by[real[0]] if m.endswith(fp_hint)), None) if real else None
if fp_mod is None:
    sys.exit(f"no module matching '{fp_hint}' to use as an input fingerprint")

groups = collections.defaultdict(list)
for p in real:
    if fp_mod in by[p]:
        groups[by[p][fp_mod]].append(p)

print(f"\n  grouping passes by '{fp_mod}' (same hash == same input):")
for h, ps in groups.items():
    print(f"    {h[:12]}  passes {ps}")

cand = [ps for ps in groups.values() if len(ps) > 1]
if not cand:
    print("\n  no two passes share an input fingerprint -- cannot compare like with like.")
    sys.exit(0)

for ps in cand:
    print(f"\n  === passes {ps} (identical {fp_mod.split('.')[-1]}) ===")
    mods = sorted({m for p in ps for m in by[p]}, key=order)
    first = None
    for m in mods:
        hs = [by[p].get(m) for p in ps]
        if any(h is None for h in hs):
            continue
        differs = len(set(hs)) > 1
        if differs and first is None:
            first = m
        mark = "DIFFERS" if differs else "same"
        # print EVERY module, not just up to the first difference: modules that are
        # identical AFTER the first difference are informative too (a clean ple_embedding
        # sitting inside a differing layer is what separates "the lookup" from "the layer").
        print(f"    {m:<52}{' '.join(h[:10] for h in hs)}   {mark}")
    print(f"\n    FIRST DIFFERENCE: {first or 'none -- all modules identical across these passes'}")
    if first and first in races:
        print("    ^ but this module is also RACE-flagged, so the difference may be the async"
              "\n      write rather than the computation.")
