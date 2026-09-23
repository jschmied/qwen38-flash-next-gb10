# I3c: sustained diverse traffic for D minutes at concurrency C; samples memory every 10 s.
import json, sys, time, random, threading, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
D = float(sys.argv[1]); C = int(sys.argv[2]); corpus = open("/opt/llm/runners/corpus_frozen.txt").read()
stop = time.time() + 60 * D; stats = dict(ok=0, err=0, prompt_tokens=0, completion_tokens=0); lock = threading.Lock()
def worker(seed):
    rng = random.Random(seed)
    while time.time() < stop:
        n = rng.choice([3000, 12000, 30000, 60000]); o = rng.randrange(0, len(corpus) - n)
        b = json.dumps({"model": "flashnext", "temperature": 0.7, "max_tokens": rng.choice([64, 256, 512]),
                        "messages": [{"role": "user", "content": f"[{seed}-{o}] " + corpus[o:o + n] + "\nSummarise."}],
                        "chat_template_kwargs": {"enable_thinking": False}}).encode()
        try:
            r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
            u = json.loads(urllib.request.urlopen(r, timeout=900).read())["usage"]
            with lock: stats["ok"] += 1; stats["prompt_tokens"] += u["prompt_tokens"]; stats["completion_tokens"] += u["completion_tokens"]
        except Exception as e:
            with lock: stats["err"] += 1
samples = []
def sampler():
    while time.time() < stop:
        m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemAvailable", "SwapTotal", "SwapFree")}
        p = open("/proc/pressure/memory").read().split("\n")[1].split()[1].split("=")[1]
        samples.append((round(m["MemAvailable"] / 2**20, 1), round((m["SwapTotal"] - m["SwapFree"]) / 2**20, 1), float(p)))
        time.sleep(10)
th = [threading.Thread(target=worker, args=(i,)) for i in range(C)] + [threading.Thread(target=sampler)]
[t.start() for t in th]; [t.join() for t in th]
av = [s[0] for s in samples]; sw = [s[1] for s in samples]; ps = [s[2] for s in samples]
print(json.dumps({"minutes": D, "concurrency": C, **stats, "memavail_min_gib": min(av), "swap_max_gib": max(sw),
                  "psi_full_avg10_max": max(ps), "psi_full_avg10_p90": sorted(ps)[int(0.9 * len(ps))], "samples": len(samples)}))
