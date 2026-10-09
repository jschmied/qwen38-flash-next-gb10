import re, sys
KEY = re.compile(r"((?:qmmf|reduce|prompt|fp8_expert|fp8_expert_prompt|nvfp4_expert)_kernelI.*?EE)")
def kernels(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            k = KEY.search(m.group(1)); cur = k.group(1) if k else None
            if cur: out[cur] = []
            continue
        if cur and re.match(r"\s*(/\*[0-9a-f]{4}\*/|/\* 0x)", line):
            out[cur].append(re.sub(r"\s+", " ", line.strip()))
    return out
py, zg = kernels(sys.argv[1]), kernels(sys.argv[2])
both = sorted(set(py) & set(zg)); eq = [k for k in both if py[k] == zg[k]]
print(f"{sys.argv[3]}: {len(eq)}/{len(both)} SASS-equal; zig-only {len(set(zg)-set(py))}")
for k in both:
    if py[k] != zg[k]: print("   DIFFER", k[:70], len(py[k]), len(zg[k]))
