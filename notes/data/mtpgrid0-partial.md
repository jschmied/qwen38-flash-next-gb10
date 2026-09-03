# mtpgrid0 — unpatched stock stack, EOS-correct 8-turn loop, batch 4096 (2026-09-03)
Raw file lost with the /tmp scratchpad in the 17:08 reboot; values transcribed from the monitor events.
Start a complete (n=0..8), start b through n=6; start c not run.

| n | start a: rate / mean_accept_len / s per turn / tokens | start b |
| --- | --- | --- |
| 0 | — / — / 2.09 / 231 | — / — / 1.85 / 203 |
| 1 | 72.3 % / 1.72 / 2.44 / 197 | 72.7 % / 1.73 / 2.41 / 213 |
| 2 | 64.0 % / 2.28 / 2.40 / 229 | 65.3 % / 2.31 / 2.19 / 144 |
| 3 | 44.8 % / 2.35 / 2.10 / 125 | 50.2 % / 2.51 / 2.42 / 210 |
| 4 | 46.6 % / 2.86 / 2.56 / 223 | 48.8 % / 2.95 / 2.75 / 241 |
| 5 | 37.4 % / 2.87 / 2.48 / 192 | 40.9 % / 3.05 / 2.18 / 122 |
| 6 | 38.5 % / 3.31 / 2.66 / 248 | 29.7 % / 2.78 / 2.47 / 145 |
| 7 | 27.4 % / 2.92 / 2.88 / 210 | (not run) |
| 8 | 25.2 % / 3.01 / 2.98 / 228 | (not run) |

Fixed stack (finding 61, 3 starts, identical to the decimal): n=1 74 % / 1.74; n=2 66 % / 2.32; n=3 57 % / 2.71;
n=4 41 % / 2.64; n=5 34 % / 2.71; n=6 36 % / 3.17; n=7 30 % / 3.12; n=8 25 % / 2.98.
