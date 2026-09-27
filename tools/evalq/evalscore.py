#!/usr/bin/env python3
"""Score evalq runs offline and compare them item by item. Run as an unprivileged user (never root): HumanEval
completions are executed, each in a fresh python process with a 10 s timeout, a 2 GiB address-space limit and
an empty temp cwd. argv: <results dir> <tag> [<tag> ...]. Prints per-tag scores and, for every pair of tags,
the discordant items (a right / b wrong and the reverse) plus an exact McNemar two-sided p."""
import gzip, itertools, json, math, os, re, resource, subprocess, sys, tempfile
HE = {json.loads(l)["task_id"]: json.loads(l) for l in gzip.open(os.path.join(os.path.dirname(__file__),
                                                                              "HumanEval.jsonl.gz"), "rt")}


def code_of(text, prompt):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    body = max(blocks, key=len) if blocks else text
    # the model restates the function; keep the prompt's imports/helpers ahead of it
    return prompt + "\n" + body if f"def " not in body else "\n".join(
        l for l in prompt.splitlines() if l.startswith(("import ", "from "))) + "\n" + body


def limit():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def run_he(p, text):
    src = code_of(text, p["prompt"]) + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-c", src], cwd=d, capture_output=True, timeout=10,
                               preexec_fn=limit, env={"PATH": "/usr/bin:/bin"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


assert os.geteuid() != 0, "run as an unprivileged user"
res, tags = sys.argv[1], sys.argv[2:]
score = {}
for t in tags:
    s = {}
    for l in open(f"{res}/evalq-{t}.jsonl"):
        r = json.loads(l)
        if r["task"] == "gsm8k":
            s[("gsm8k", r["i"])] = r["ok"]
        else:
            s[("humaneval", r["i"])] = run_he(HE[r["task_id"]], r["text"])
    score[t] = s
    for task in ("gsm8k", "humaneval"):
        v = [ok for (k, _), ok in s.items() if k == task]
        print(f"{t:12s} {task:9s} {sum(v)}/{len(v)} = {100 * sum(v) / max(1, len(v)):.2f} %")
for a, b in itertools.combinations(tags, 2):
    for task in ("gsm8k", "humaneval"):
        keys = [k for k in score[a] if k[0] == task and k in score[b]]
        ab = sum(score[a][k] and not score[b][k] for k in keys)
        ba = sum(score[b][k] and not score[a][k] for k in keys)
        print(f"{a} vs {b} {task:9s}: {a} only {ab}, {b} only {ba}, Δ {100 * (ba - ab) / max(1, len(keys)):+.2f} pp, "
              f"McNemar p={mcnemar(ab, ba):.3f}")
