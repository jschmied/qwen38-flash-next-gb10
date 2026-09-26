# Item 7: vm.compaction_proactiveness 20 (default) vs 0, written 2026-09-26 ~15:40, before the run
Source: bilikaz/qwen38-flash-next-recipe (4-5 s stall every ~37 s, ~-10 %). Box counters since boot: compact_stall
4.25 M, kcompactd scanned 2.05e10 pages. Arms (clone venv, prod-like, RecoverSSM on, KV 4 GiB) set the sysctl in
the launcher wrapper; the driver restores 20 at the end. Probe: decprobe c=1/c=4 + a 4-min sustained c=4 run in 5 s
windows, compaction counters before/after. 2 starts, alternating.
- H1: sustained c=4 mean throughput +2..+10 % with 0, and fewer dip windows (< 70 % of median); p10 up.
- H2: c=1 decprobe within +-1.5 % (short runs mostly miss a 37 s period).
- H3: compact_daemon_* deltas ~0 in the 0 arm, > 0 in the 20 arm (proves the mechanism ran); outputs identical.
- Outside: no difference and no daemon activity in the 20 arm either -> the stall needs their memory edge
  (27 GB KV, ~3 GB free); ours at KV 4 GiB may not reach it.
