import json,glob,os,hashlib,random
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/opt/llm/models/qwen38-flash-next-fp8head"); random.seed(0)
fs=sorted(glob.glob('/opt/llm/swebench-runs/**/*.traj.json',recursive=True)); out=[]
for f in fs:
    d=json.load(open(f)); iid=d.get("instance_id",os.path.basename(f))
    if int(hashlib.md5(iid.encode()).hexdigest(),16)%10!=0: continue
    m=d["messages"]; sysmsg=next((x["content"] for x in m if x["role"]=="system"),"")
    aidx=[i for i,x in enumerate(m) if x["role"]=="assistant"]
    for cut in random.sample(aidx[4:], min(3,len(aidx[4:]))):
        lines=[]
        for x in m[1:cut]:
            r=x["role"]; c=x.get("content") or ""
            if r=="assistant":
                tc=x.get("tool_calls") or []
                cmd=""
                for t in tc:
                    try: cmd=json.loads(t["function"]["arguments"]).get("command","")
                    except Exception: cmd=t["function"]["arguments"]
                lines.append(f"ASSISTANT:\n{(x.get('reasoning_content') or '')[:600]}\n```bash\n{cmd}\n```")
            elif r=="tool": lines.append(f"TOOL OUTPUT:\n{c[:3000]}")
            else: lines.append(f"USER:\n{c}")
        body="\n\n".join(lines); ids=tok(body,add_special_tokens=False)["input_ids"]
        if len(ids)>5000: body=tok.decode(ids[-5000:])
        prompt=sysmsg[:1500]+"\n\nHere is the transcript so far:\n\n"+body+"\n\nContinue: think through the next step and give the next bash command."
        out.append({"instance":iid,"cut":cut,"tokens":len(tok(prompt,add_special_tokens=False)["input_ids"]),"prompt":prompt})
json.dump(out,open("/tmp/claude-1000/-home-jschmied-git-dgx-spark-setup-guide/c5ecde82-2840-4944-bce7-68e07b289e98/scratchpad/dv/dv_prompts.json","w"))
print(len(out),"prompts; tokens min/median/max", min(o["tokens"] for o in out), sorted(o["tokens"] for o in out)[len(out)//2], max(o["tokens"] for o in out))
