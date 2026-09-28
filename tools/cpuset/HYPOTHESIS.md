CPUSET (agenda item 8, from bilikaz's recipe: pin to the X925 cores 5-9,15-19, they report +2-3 %), 2026-09-28 ~13:35.
Box check: CPUs 5-9 and 15-19 are Cortex-X925 (MIDR d85, 3.9 GHz max); 0-4 and 10-14 are A725 (d87, 2.8 GHz).
Arms on prod's venv and flags (prodval2 config + KV 4 GiB): pinned (launcher wrapper `taskset -c 5-9,15-19`, whole
process tree) vs free, 2 starts each, alternating; probe nvprobe (decode c=1 code/prose, c=4, TTFT 8k/30k, replay).
H: c=1 decode -0..-3 % (the host is on the critical path at c=1: scheduling, sampling, launches; 54 % GPU duty cycle);
TTFT -0..-2 %; c=4 -0..-2 %. Hashes identical (CPU placement cannot change GPU numerics).
Out of range: any cell slower by > 1 % (10 cores for the whole tree causes contention), or a gain > 5 % (then check
what else moved). Persisting a pin in prod would be a launcher change -> needs the user's go.
