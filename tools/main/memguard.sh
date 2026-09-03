#!/bin/bash
# PSI-based memory guard v2 (2026-09-03 20:57): stop any fx-* unit only on a real thrash —
# memory "full avg10" > 60 % AND MemAvailable < 3 GiB for 3 consecutive 5 s checks.
# v1 used "some avg10" alone and killed a healthy load (S1_c, 20:54: some=67.7 %, 28.8 GiB avail)
# because paging to /swapfile-fnext during the load phase raises "some" without any stall.
OUT=/opt/llm/runners/results/memguard.txt; hi=0
while :; do
  u=$(systemctl list-units --no-legend 'fx-*' | awk '{print $1}' | head -1); [ -n "$u" ] || { sleep 10; continue; }
  f=$(awk '/^full/ {split($2,a,"="); print a[2]}' /proc/pressure/memory); a=$(free -m | awk 'NR==2{print $7}')
  if awk "BEGIN{exit !($f > 60 && $a < 3072)}"; then hi=$((hi+1)); else hi=0; fi
  if [ $hi -ge 3 ]; then echo "MEMGUARD $(date +%H:%M:%S): psi full avg10=$f avail=${a}MiB for 3 checks -> stopping $u" >>$OUT; systemctl stop $u; hi=0; fi
  sleep 5
done
