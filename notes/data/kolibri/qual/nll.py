import numpy as np, sys
V=128000
ids=[int(t) for t in open(sys.argv[1]).read().replace(',',' ').split()][:2048]
for f in sys.argv[2:]:
    a=np.memmap(f,dtype=np.float32,mode='r').reshape(-1,V); n=a.shape[0]-1
    nll=[]; acc=0
    for i in range(0,n,128):
        x=a[i:min(i+128,n)].astype(np.float64); t=np.array(ids[i+1:i+1+x.shape[0]])
        m=x.max(1,keepdims=True); lse=(m[:,0]+np.log(np.exp(x-m).sum(1)))
        nll+=list(lse-x[np.arange(len(t)),t]); acc+=(x.argmax(1)==t).sum()
    nll=np.array(nll)
    print(f"{f.split('/')[-1]}: NLL mean {nll.mean():.4f} (ppl {np.exp(nll.mean()):.3f}), next-token top-1 {100*acc/n:.2f} %, rows {n}")
