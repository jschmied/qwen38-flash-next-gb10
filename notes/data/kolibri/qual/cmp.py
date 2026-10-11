import numpy as np, sys
V=128000
a=np.memmap(sys.argv[1],dtype=np.float32,mode='r').reshape(-1,V)
b=np.memmap(sys.argv[2],dtype=np.float32,mode='r').reshape(-1,V)
n=a.shape[0]; top=0; kl=[]; maxd=[]
for i in range(0,n,128):
    x=a[i:i+128].astype(np.float64); y=b[i:i+128].astype(np.float64)
    top+=(x.argmax(1)==y.argmax(1)).sum()
    px=x-x.max(1,keepdims=True); px=px-np.log(np.exp(px).sum(1,keepdims=True))
    py=y-y.max(1,keepdims=True); py=py-np.log(np.exp(py).sum(1,keepdims=True))
    kl+=list((np.exp(px)*(px-py)).sum(1)); maxd+=list(np.abs(x-y).max(1))
kl=np.array(kl); maxd=np.array(maxd)
print(f"rows {n}: top-1 equal {top}/{n} ({100*top/n:.2f} %), KL mean {kl.mean():.2e} p99 {np.percentile(kl,99):.2e} max {kl.max():.2e}, |dlogit| median {np.median(maxd):.3f} max {maxd.max():.3f}")
