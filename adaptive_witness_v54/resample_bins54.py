"""Paired catalog bootstrap with relearning in both discovery halves; exploratory intervals."""
import json
import os
import numpy as np
from sign_bins54 import ROOT,N,G,GROUPS,read_panel

def main():
    assert os.environ.get('SLURM_JOB_ID')
    panels=[read_panel(i) for i in range(N)]
    draws=np.random.default_rng(5401217).integers(0,6,(2000,2,6))
    mult=np.stack([np.bincount(x.ravel(),minlength=12) for x in draws+np.array([0,6])[None,:,None]])
    mult=np.vstack([np.ones(N),mult])
    n=np.zeros((12,6))
    sums=np.zeros((12,6,2))
    cdfs={}
    for i,(risk,raw,groups) in enumerate(panels):
        groups=np.broadcast_to(groups[None],risk.shape[:-1])
        for e,event_weight in enumerate([1-G.mean(),G.mean()]):
            for b in range(6):
                select=groups[:,e]==b
                norm=select.size
                n[i,b]+=event_weight*select.mean()
                sums[i,b]+=event_weight*risk[:,e][select].sum(axis=0)/norm
                cdfs[i,e,b]=[np.sort(raw[:,e,...,j][select]) for j in (0,1)]
    delta=sums[...,1]-sums[...,0]
    rules=np.concatenate([np.repeat((mult[:,6:]@delta[6:]<0)[:,None,:],6,axis=1),
                          np.repeat((mult[:,:6]@delta[:6]<0)[:,None,:],6,axis=1)],axis=1)
    coeff=mult[:,:,None]/12
    mass=(coeff*rules*n).sum((1,2))
    fixed=(coeff[...,None]*sums).sum((1,2))
    mu=(coeff*np.where(rules,sums[...,1],sums[...,0])).sum((1,2))
    mus=np.column_stack([mu,fixed,(1-mass)*fixed[:,0]+mass*fixed[:,1]])
    prob=[]
    for arm in ['learned','R','I','random']:
        # p90 of the session distribution; reference and impact outcome noise paired.
        mix=rules if arm=='learned' else np.broadcast_to({'I':1.,'R':0.,'random':mass[:,None,None]}.get(arm,0.),rules.shape)
        lo,hi=np.full(len(mult),0.),np.full(len(mult),1.)
        for _ in range(28):
            mid=(lo+hi)/2;cdf=np.zeros_like(mid)
            for (i,e,b),d in cdfs.items():
                w=(1-G.mean() if e==0 else G.mean()) / (3*5*4*256)
                for j in (0,1):
                    choice=mix[:,i,b] if j==1 else 1.-mix[:,i,b]
                    cdf+=w*coeff[:,i,0]*choice*np.searchsorted(d[j],mid,'right')
            high=cdf>=.9;hi=np.where(high,mid,hi);lo=np.where(high,lo,mid)
        prob.append(hi)
    qs=np.stack(prob,axis=1)
    out={'cols':['twofold','reference','impact','rate_matched_random'],'bootstrap':2000,'actual_mean':mus[0].tolist(),'actual_p90':qs[0].tolist()}
    for name,v in [('mean',mus),('p90',qs)]:
        out[name+'_delta']=(v[0,0]-v[0,1:]).tolist()
        out[name+'_delta_95']=np.quantile(v[1:,0,None]-v[1:,1:],[.025,.975],axis=0).T.tolist()
    (ROOT/'resample_bins54.json').write_text(json.dumps(out,indent=2))
    print(out)
if __name__=='__main__':main()
