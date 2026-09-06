# Reconstruct the semaphore protocol from a server log with PLESEM/PLEWAIT/PLEFLAG/PLECHECK lines (log order).
import re, sys
rx=re.compile(r"\((\w+) pid=(\d+)\).*?(PLESEM (\w+) pid=\d+ t=([\d.]+) stream=(\S+) sem=(\S+) capturing=(\w+) via (\S+)|PLEWAIT flag_before_wait=(\d) stream=(\S+) tokens=(\d+)|PLEFLAG (\w+) tokens=(\d+) flags=\{[^}]*: (\d)\}|PLECHECK tokens=(\d+) layer=\S*layers\.1\.\S+ nonzero_rows=(\d+)|(Running FlashInfer autotune[^\n]*|Initial profiling/warmup[^\n]*|init engine[^\n]*))")
for line in open(sys.argv[1], errors="replace"):
    m=rx.search(line)
    if not m: continue
    proc, pid = m.group(1), m.group(2)
    if m.group(3) and m.group(3).startswith("PLESEM"):
        print(f"{proc:16s} SEM {m.group(4):10s} t={m.group(5):>8s} stream={m.group(6)} cap={m.group(8)} via {m.group(9)}")
    elif m.group(10) is not None:
        print(f"{proc:16s} GPUWAIT sees flag={m.group(10)} stream={m.group(11)} tokens={m.group(12)}")
    elif m.group(13):
        print(f"{proc:16s} FLAG {m.group(13):14s} tokens={m.group(14)} flag={m.group(15)}")
    elif m.group(16):
        print(f"{proc:16s} BUFFER tokens={m.group(16)} nonzero_rows={m.group(17)}")
    elif m.group(18):
        print(f"{proc:16s} --- {m.group(18)[:90]}")
