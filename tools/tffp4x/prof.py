#!/usr/bin/env python3
"""Where an arm's time goes: one 8k prefill and one 256-token decode under torch.profiler (CUDA kernels by name), plus
the generate() stats (drafting). argv: <tensorfold src> <model dir> <mode>"""
import collections, json, sys, sysconfig
from pathlib import Path

src, model, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from torch.profiler import ProfilerActivity, profile                             # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

precision.set_mode(mode, asked=True)
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))
eng = FlashNextEngine(model, max_len=16384)
eng.generate(tok.encode("# warm\n") + ids[:1024], 1, None, lambda t: None)
chat = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Write a Python function that parses ISO 8601 "
                  "durations like P3DT4H5M into seconds, with tests."}], tokenize=False, add_generation_prompt=True,
                  enable_thinking=False), add_special_tokens=False)
eng.generate(chat, 64, None, lambda t: None, stop_eos=False)
torch.cuda.synchronize()


def table(prof, label):
    agg = collections.defaultdict(lambda: [0.0, 0])
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            a = agg[ev.name[:90]]
            a[0] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
            a[1] += 1
    total = sum(v[0] for v in agg.values())
    print(f"== {label}: GPU kernel time {total / 1e3:.1f} ms", flush=True)
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"  {t / 1e3:9.2f} ms {100 * t / total:5.1f} % x{n:6d}  {name}", flush=True)


with profile(activities=[ProfilerActivity.CUDA]) as p:
    st = eng.generate(tok.encode("# prompt of 8192 tokens\n") + ids[:8192], 1, None, lambda t: None, stop_eos=False)
    torch.cuda.synchronize()
table(p, f"{mode} 8k prefill ({st.get('prefill_s')} s)")
with profile(activities=[ProfilerActivity.CUDA]) as p:
    st = eng.generate(chat, 256, None, lambda t: None, stop_eos=False)
    torch.cuda.synchronize()
table(p, f"{mode} decode 256")
print("STATS", json.dumps({k: v for k, v in st.items() if isinstance(v, (int, float, str))}), flush=True)
