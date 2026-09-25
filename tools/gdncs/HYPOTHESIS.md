# FNGDNCS: streaming stores for the GDN spec-decode state snapshots (written 2026-09-25 ~19:05, before the run)
Speed-of-light 4o: the GDN update writes 4 fp32 state snapshots (12 MiB) per layer per step in 42 us, faster than DRAM;
they sit dirty in L2 and their write-back costs the next kernels ~40 us/layer (out_proj 70->100 us, mixer 31->47 us),
~1.5 ms/step. `.cs` stores drain them during the GDN kernel instead. Same DRAM bytes, less interference.
A/B base vs cs, KV 4 GiB, decprobe twice per start (warm second pass measured), 2 starts per arm.
- H: c=1 ms per verify cycle -0.3 to -1.2 ms (-0.5 to -2 %) in every start pairing; c=4 similar; output hashes
  identical (values unchanged).
- Outside: cs slower -> the GDN kernel itself becomes DRAM-bound and pays more than the aftershock saved; then the
  only lever is fewer bytes (ReplaySSM #49887 or fewer/narrower snapshots).
