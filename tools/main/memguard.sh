#!/bin/bash
# PSI-based memory guard: stop any fx-* unit if memory "some avg10" stays > 60 % for 3 consecutive 5 s checks.
OUT=/opt/llm/runners/results/memguard.txt; hi=0
while :; do
  u=$(systemctl list-units --no-legend 'fx-*' | awk '{print $1}' | head -1); [ -n "$u" ] || { sleep 10; continue; }
  p=$(awk '/^some/ {split($2,a,"="); print a[2]}' /proc/pressure/memory); a=$(free -m | awk 'NR==2{print $7}')
  if awk "BEGIN{exit !($p > 60)}"; then hi=$((hi+1)); else hi=0; fi
  if [ $hi -ge 3 ]; then echo "MEMGUARD $(date +%H:%M:%S): psi some avg10=$p avail=${a}MiB for 3 checks -> stopping $u" >>$OUT; systemctl stop $u; hi=0; fi
  sleep 5
done
