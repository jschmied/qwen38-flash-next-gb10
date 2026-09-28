"""Code-provenance scan: for each TensorFold CUDA/Triton file, the share of its token shingles found verbatim in each
donor corpus (exact identifiers, k=12) and with identifiers normalized (k=20). Comments and whitespace stripped."""
import os, re, sys, json, glob, collections
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][-+]?\d+)?[fF]?|==|!=|<=|>=|<<|>>|&&|\|\||\+=|-=|\*=|/=|->|::|\S")
KW = set("""if else for while return def class import from as in not and or is None True False const int float void auto
unsigned bool struct template typename static inline constexpr __global__ __device__ __forceinline__ tl triton jit
range static_range pragma unroll include namespace using break continue lambda with yield half float4 uint32_t int32_t
int64_t uint8_t size_t""".split())
def strip(src, ext):
    if ext in (".py",):
        src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', " ", src); src = re.sub(r"#[^\n]*", " ", src)
    else:
        src = re.sub(r"/\*[\s\S]*?\*/", " ", src); src = re.sub(r"//[^\n]*", " ", src)
    return src
def toks(path):
    ext = os.path.splitext(path)[1]
    try: s = open(path, errors="ignore").read()
    except Exception: return [], []
    t = TOK.findall(strip(s, ext)); n = [x if (x in KW or not re.match(r"[A-Za-z_]", x)) else "ID" for x in t]
    return t, n
def shingles(t, k): return {tuple(t[i:i+k]) for i in range(max(0, len(t)-k+1))}
def corpus(files):
    e, n = set(), set()
    for f in files:
        t, m = toks(f); e |= shingles(t, 12); n |= shingles(m, 20)
    return e, n
EXT = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".metal")
def walk(root):
    out = []
    for dp, dn, fn in os.walk(root):
        if "/.git" in dp: continue
        out += [os.path.join(dp, f) for f in fn if f.endswith(EXT)]
    return out


import sys, os, re
f, roots = sys.argv[1], sys.argv[2:]
t, _ = toks(f); K = 12
idx = {}
for r in roots:
    for g in walk(r):
        u, _ = toks(g)
        for i in range(max(0, len(u)-K+1)): idx.setdefault(tuple(u[i:i+K]), g)
runs, i = [], 0
while i < len(t)-K+1:
    key = tuple(t[i:i+K])
    if key in idx:
        j = i
        while j < len(t)-K+1 and tuple(t[j:j+K]) in idx: j += 1
        runs.append((j - i + K - 1, i, idx[key])); i = j
    else: i += 1
runs.sort(reverse=True)
for n, i, g in runs[:6]:
    print(n, "tokens |", g.split("scratchpad/tfold/")[-1].split("site-packages/")[-1][-70:], "|", " ".join(t[i:i+min(n,40)])[:230])
