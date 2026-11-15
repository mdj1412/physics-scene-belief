import torch
from model import *
from mixture import mv
from acquisition import plan
def prepare(b):
    b.features={a:torch.cat([y,(y[:,:,None]*y[:,None,:]).flatten(1)],-1) for a,y in b.white.items() if a}
    return b
def native(b,w,z,sizes=SIZES):
    zm=w@z;scores=[torch.zeros(len(sizes),device=DEV,dtype=DT)]
    for a in [1,2,3]:
        y=b.white[a];D=y.shape[1];mom=w@b.features[a];ym=mom[:D]
        S=mom[D:].reshape(D,D)-ym[:,None]*ym[None,:]+torch.eye(D,device=DEV,dtype=DT)
        C=torch.stack([(w*z[:,j])@y for j in range(z.shape[1])])-zm[:,None]*ym[None,:]
        scores.append(group((C*torch.linalg.solve(S,C.T).T).sum(-1),sizes))
    return torch.stack(scores)
def nested(b,w,z,o,ow,oz,preplan,sizes=SIZES):
    u,eps=preplan;M=len(u);ix=torch.searchsorted(ow.cumsum(0),u.contiguous()).clamp_max(o.N-1);truth=oz[ix]
    base=group((truth-w@z)**2,sizes).mean(0);rs=[base];rs128=[base];lp=w.clamp_min(1e-300).log()
    for a in [1,2,3]:
        vals=[]
        for start in range(0,M,128):
            idx=ix[start:start+128];y=o.mu[a][idx]+eps[start:start+128,:o.mu[a].shape[-1]]*noise(a)
            post=torch.softmax(lp+b.ll(a,y),-1);pred=mv(post,z);vals.append(group((pred-truth[start:start+128])**2,sizes))
        vals=torch.cat(vals);rs.append(vals.mean(0));rs128.append(vals[:128].mean(0))
    return torch.stack(rs),torch.stack(rs128)
