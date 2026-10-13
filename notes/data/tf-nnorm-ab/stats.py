import json, sys, numpy as np, torch
V=131072
def load(p):
    a=np.fromfile(p,dtype=np.uint16).astype(np.uint32)<<16
    return torch.from_numpy(a.view(np.float32).reshape(-1,V)).double()
A=load(sys.argv[1]); B=load(sys.argv[2]); off=0
print(sys.argv[2], 'identical rows', int((A==B).all(1).sum()), 'of', A.shape[0])
T=dict(n=0,dn=0,ag=0,kl=0,fl=0,big=0)
for n in ['docs','code','german','chat']:
    toks=json.load(open(f'teach-{n}.json'))['tokens']; L=len(toks)
    a=A[off:off+L-1]; b=B[off:off+L-1]; want=torch.tensor(toks[1:]); off+=L
    la=torch.log_softmax(a,1); lb=torch.log_softmax(b,1)
    dn=(-lb[range(L-1),want]+la[range(L-1),want]).mean().item()
    kl=(la.exp()*(la-lb)).sum(1); ag=(a.argmax(1)==b.argmax(1)).float().mean().item()*100
    fl=[(a[i,a[i].argmax()]-a[i,b[i].argmax()]).item() for i in range(L-1) if a[i].argmax()!=b[i].argmax()]
    print(f"  {n}: dNLL {dn:+.4f} argmax {ag:.2f} % KL mean {kl.mean():.5f} max {kl.max():.4f} flips {len(fl)} largest {max(fl) if fl else 0:.3f}")
    T['n']+=1; T['dn']+=dn; T['ag']+=ag; T['kl']+=kl.mean().item(); T['big']=max(T['big'], max(fl) if fl else 0)
print(f"  mean: dNLL {T['dn']/4:+.4f} argmax {T['ag']/4:.2f} % KL {T['kl']/4:.5f} largest flip {T['big']:.3f}")
