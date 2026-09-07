# GB10 / DGX Spark: USB storage, PCIe tunneling, and the PD throttle wedge

Written 2026-09-07 after asking whether an external drive could hold the ~240 GB of 27B quant
variants that dominate `/opt/llm/models`. Short answer: **an external NVMe will not behave the way
the port speed suggests, and the more important finding was a GPU failure mode worth checking on
this box regularly.**

## 1. There is no PCIe tunneling on this platform

Checked on our own box before consulting anyone:

| check | result |
| --- | --- |
| `/sys/bus/thunderbolt/devices/` | empty — no domain, no host router |
| `thunderbolt` module | present in the tree, **not loaded at boot**; loading it registers the ACPI bus type but still enumerates no host router |
| kernel config | `CONFIG_USB4=m`, `CONFIG_USB4_NET=m` — the kernel supports it |
| root hubs | 6 × `bcdUSB=3.10` @ **20000 Mb/s** + 6 × `bcdUSB=2.00` @ 480 Mb/s = **6 physical ports** |
| `lspci` | no USB controller — it is on-SoC |

**A machine that supports PCIe tunneling enumerates a host router with nothing plugged in.** This one
does not, so the test needs no purchase.

Community measurements agree ([NVIDIA forum thread](https://forums.developer.nvidia.com/t/dgx-spark-usb-ports-are-usb-4-40gbps-o-so-why/349121)):
USB4/TB NVMe enclosures enumerate as **`/dev/sdZ` (USB mass storage), never `/dev/nvmeXnY`** — the
direct test of tunneling, and it fails. NVIDIA does not publish the USB spec; the ASUS Ascent GX10
datasheet (same platform) states USB 3.2 Gen 2×2.

**40 Gb/s and 20 Gb/s are both true and describe different layers.** USB4 negotiates a 40 Gb/s *link*
that carries tunneled protocols; the USB 3.2 Gen 2×2 tunnel inside it is 20 Gb/s, which is what the
xHCI root hubs report. Since there is no host router here, the 20 Gb/s path is the **only** path.

## 2. The sustained-write problem, and why "thermal" is probably the wrong label

Reported: reads fall from ~2 GB/s to **<40 MB/s within minutes**, and **do not recover until a
physical replug or reboot**.

That last property is the interesting one. Real thermal throttling recovers when the part cools.
Something that stays degraded until unplugged looks like **link or protocol degradation** — UAS
timeouts falling back to BOT, or the link renegotiating to a lower generation and staying there. The
supporting evidence fits: the stable configuration users found was **disabling UAS plus a fan**, and
there is a separate documented bug where
[ASM2464PD enclosures come up at USB 2.0 480 Mbps on every power-on](https://forums.developer.nvidia.com/t/asm2464pd-usb-c-3-2-2x2-enclosure-always-fall-back-to-usb2-0-480-mbps-soft-replug/380009)
until replugged. **No temperature readings were posted anywhere in that thread** — "thermal" is an
attribution, not a measurement. Any thermal component is the *enclosure* (bridge chip + NVMe sealed
in metal with no airflow), not the Spark.

Enclosure silicon matters more than the marketing number: **ASM2464PD ≈ 1.8–2 GB/s**, **JHL7440 ≈ 950
MB/s** despite "40 Gbps" on the box.

**If we ever buy one:** ASM2464PD, a fan on the enclosure, `usb-storage.quirks` to disable UAS, and
validate with a **sustained multi-hundred-GB write**, not a burst benchmark — the sustained case is
both the one we need (moving ~240 GB) and the one that breaks.

## 3. The PD throttle wedge — check this on OUR box, it invalidates benchmarks

Found while researching the above and far more relevant to us
([GX10_PD_Throttle_Fix](https://github.com/Sggin1/DGX-SPARK/blob/main/GX10_PD_Throttle_Fix.md),
[spark-gpu-throttle-check](https://github.com/hoesing/spark-gpu-throttle-check)).

The USB-C PD controller in the power brick can lose sync with the GPU and pin it at **611 MHz drawing
~13 W**, with a misleading ~50 °C "cap" that reflects the low power state rather than heat. Reported
effect: Qwen-34B inference **61 → 30 tok/s**. Triggers: **sleep/resume**, a mid-load crash, or a
firmware update without a full power cycle — we have had box-kills and firmware updates, so this is
not hypothetical for us.

**Detection** (cheap, no load required — a wedged box cannot reach 2 GHz):

```
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv
nvidia-smi -q -d PERFORMANCE | grep -A12 "Clocks Event Reasons"
```

| | wedged | healthy | **ours, 2026-09-07 20:5x** |
| --- | --- | --- | --- |
| SM clock | 611 MHz | 2200–2600 | **2418 MHz** (max 3003) |
| power under load | ~13 W | 80–100 W | 13.35 W **at 1 % util — idle, not wedged** |
| throttle reasons | — | none | **all Not Active** |

⚠️ **Read the clock, not the wattage.** 13 W looks like the wedge signature but is normal at idle;
the discriminator is the 611 MHz pin. Confirm under real load when a benchmark is running.

**Fix if wedged:** full cold drain — shut down, unplug the 240 W PSU from wall *and* unit, remove all
USB-C peripherals, hold power ~30 s, wait 60 s+, replug PSU only. Then flash the latest PD firmware
via `fwupdmgr` and cold-drain a **second** time so it loads from EEPROM.

## Why this is in the repo

Every throughput number we publish assumes the GPU is not wedged. We have never checked, and the
trigger list (sleep/resume, mid-load crash, firmware update) matches events in this box's history.
Add the clock check to the benchmark preflight rather than trusting that a run is comparable to one
from last week.
