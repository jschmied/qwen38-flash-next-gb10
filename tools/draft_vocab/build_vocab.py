# Frequency-ranked draft vocabulary for the Flash-Next MTP head from OUR agent output distribution.
# Corpus: (A) model's own outputs from the SWE-bench trajectories (reasoning + tool-call args + content) x20,
#         (B) tool outputs the model read x1, (C) code/doc files from local repos x3. Held-out: 10% of trajectories by id hash.
import json, glob, os, sys, hashlib, collections, time
from transformers import AutoTokenizer
MD="/opt/llm/models/qwen38-flash-next-fp8head"; OUT=sys.argv[1]
tok=AutoTokenizer.from_pretrained(MD); VS=len(tok)
def enc(text): return tok(text, add_special_tokens=False)["input_ids"]
def count_texts(texts, cnt, w):
    n=0
    for t in texts:
        if not t: continue
        for i in range(0,len(t),1<<20):
            ids=enc(t[i:i+(1<<20)]); n+=len(ids)
            for x in ids: cnt[x]+=w
    return n
train=collections.Counter(); held=collections.Counter(); ntr=nho=0
fs=sorted(glob.glob('/opt/llm/swebench-runs/**/*.traj.json',recursive=True)); nh=0
t0=time.time()
for f in fs:
    d=json.load(open(f)); iid=d.get("instance_id",os.path.basename(f)); ho=int(hashlib.md5(iid.encode()).hexdigest(),16)%10==0
    outs=[]; tools=[]
    for m in d["messages"]:
        if m.get("role")=="assistant":
            if m.get("reasoning_content"): outs.append(m["reasoning_content"])
            if m.get("content"): outs.append(m["content"])
            for tc in m.get("tool_calls") or []:
                a=tc.get("function",{}).get("arguments","")
                try: a=json.loads(a); a=a.get("command") or json.dumps(a)
                except Exception: pass
                outs.append(str(a))
        elif m.get("role")=="tool": tools.append(str(m.get("content")))
    if ho: nh+=1; nho+=count_texts(outs, held, 1)
    else: ntr+=count_texts(outs, train, 20); count_texts(tools, train, 1)
print(f"trajs {len(fs)} (held-out {nh}); assistant tokens train {ntr:,} held {nho:,}; {time.time()-t0:.0f}s", flush=True)
code=[]; exts=('.py','.cu','.cuh','.h','.hpp','.cpp','.md','.sh','.json','.yaml','.yml','.toml','.ts','.js','.rs','.go','.txt','.cfg','.ini')
for root in ('/opt/llm/src/vllm-main','/home/jschmied/git/vllm-mambafix','/home/jschmied/git/qwen38-flash-next-gb10','/home/jschmied/git/dgx-spark-setup-guide'):
    for dp,dn,fn in os.walk(root):
        dn[:]=[x for x in dn if x not in ('.git','node_modules','__pycache__','build','.venv')]
        for x in fn:
            p=os.path.join(dp,x)
            if x.endswith(exts) and os.path.getsize(p)<400_000:
                try: code.append(open(p,errors='ignore').read())
                except Exception: pass
nc=count_texts(code, train, 3); print(f"code files {len(code)}, tokens {nc:,}; {time.time()-t0:.0f}s", flush=True)
special=set(tok.all_special_ids)
ranked=[i for i,_ in train.most_common() if i not in special]
tot=sum(train.values()); htot=sum(held.values())
print(f"train occurrences {tot:,} distinct {len(train):,}; held-out assistant occurrences {htot:,} distinct {len(held):,}")
for cut in (8192,16384,32768,65536,131072):
    keep=list(special)+ranked[:cut-len(special)]; ks=set(keep)
    ctr=sum(train[i] for i in keep)/tot; cho=sum(held[i] for i in keep)/max(htot,1)
    print(f"  size {cut:6d}: train coverage {100*ctr:6.2f}%   held-out assistant coverage {100*cho:6.2f}%")
    if cut in (16384,32768,65536):
        with open(f"{OUT}/draft_vocab_{cut}.txt","w") as fh: fh.write("\n".join(str(i) for i in keep)+"\n")
# non-Latin share of the kept set, for the record
for cut in (16384,32768,65536):
    keep=list(special)+ranked[:cut-len(special)]
    nl=sum(1 for i in keep if any(ord(ch)>0x24F for ch in tok.decode([i])))
    print(f"  size {cut}: {nl} kept tokens contain non-Latin chars ({100*nl/cut:.1f}%)")
print("done", OUT)
