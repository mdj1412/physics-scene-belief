"""The same orientation evidence supplied AFTER acquisition to a matched terminal recalibrator."""
from pathlib import Path
import json,numpy as np
from scipy.special import expit
from direct_head34 import FAMILIES,SCHEDULES,cue_est
from decisions34 import ROOT,OUT,sample,band,dump
def features(l,a,c):
    return np.stack((l,a,np.ones_like(l)*c,l*c,a*c),-1)
def build(f,lock):
    rows=[]; hist=[]
    for b in range(8):
        for d,law in enumerate([FAMILIES[f]+'_plus',FAMILIES[f]+'_minus']):
            for a,sc in enumerate(SCHEDULES):
                p=OUT/law/f'block_{b:03d}'/sc
                with np.load(p/'choices.npz') as ch,np.load(p/'metrics.npz') as z:
                    # Evaluate precommitted fixed4 selected observations, not both outcomes.
                    aa=(z['actions'][:,:,1]==3).astype(int)
                    h=np.take_along_axis(z['probabilities'],aa[:,:,None,None],axis=2)[:,:,0]
                    y=z['labels'].astype(float); assert np.allclose(((h-y[:,:,None])**2).mean(),z['brier'][:,1].mean())
                    l=np.log(np.clip(h,.001,.999)/(1-np.clip(h,.001,.999)))
                    ab=np.broadcast_to(aa[:,:,None],l.shape);yb=np.broadcast_to(y[:,:,None],l.shape)
                    cues={}
                    for fold in (f,3-f):
                        cues[fold]=cue_est(lock,fold,ch['paid_x'][:,:4],ch['paid_y'][:,:4],ch['paid_direction'][:,:4],sc)
                    rows.append((b,d,a,l,ab,yb,h,cues))
                    for e in range(8):
                        nbin=np.rint(l[e].ravel()*50).astype(int); xx=(nbin+400)*2+ab[e].ravel()
                        ids,u,w=np.unique(xx,return_inverse=True,return_counts=True);pos=np.bincount(u,weights=yb[e].ravel())
                        hist.append(((ids//2-400)/50,ids%2,pos,w,{fold:pc[e] for fold,pc in cues.items()},1-2*d))
    return rows,hist
def fit(h,fold,mode):
    x=np.concatenate([features(l,a,[0,c[fold],o][mode]) for l,a,y,w,c,o in h]);w=np.concatenate([v[3] for v in h]);z=np.concatenate([v[2] for v in h])
    wt=w/w.sum();y=z/w;m=wt@x;sd=np.sqrt(wt@(x-m)**2);sd[sd<1e-10]=1
    xx=np.column_stack((np.ones(len(x)),(x-m)/sd));beta=np.zeros(6);pen=np.diag([0]+[.001]*5)
    for _ in range(40):
        p=expit(xx@beta);grad=xx.T@(wt*(p-y))+pen@beta
        if np.max(abs(grad))<1e-10:break
        beta-=np.linalg.solve((xx.T*(wt*p*(1-p)))@xx+pen,beta*0+grad)
    return (m,sd,beta)
def main():
    lock=json.loads((ROOT/'diag_models.json').read_text());data=[build(f,lock) for f in range(4)]
    values=np.zeros((4,8,2,2,4));model_log={}
    for f in range(4):
        models=[fit(data[3-f][1],f,m) for m in range(3)]
        model_log[str(f)]=[dict(mean=m,sd=sd,beta=b) for m,sd,b in models]
        for b,d,a,l,ac,y,h,cs in data[f][0]:
            values[f,b,d,a,0]=((h-y)**2).mean()
            for mode,((m,sd,beta),cu) in enumerate(zip(models,[0,cs[f][:,None,None],1-2*d])):
                x=features(l,ac,cu);pr=expit(((x-m)/sd)@beta[1:]+beta[0])
                values[f,b,d,a,mode+1]=((pr-y)**2).mean()
    dump('terminal_models34.json',model_log)
    mu=values.mean((0,1,2))
    def dif(v):return np.stack([v[...,a,i]-v[...,a,j] for a in (0,1) for i,j in ((2,1),(3,1),(3,2))],-1)
    r=np.random.default_rng(340664);idx=sample(r,12000);boot=values[np.arange(4)[None,:,None],idx].mean((1,2,3))
    dd=dif(mu);dump('terminal_stats34.json',dict(means=mu,delta=dd,ci=band(dd,dif(boot)),methods=['fixed4_original','neutral','estimated','oracle'],by_family=values.mean((1,2)),training='0.02 bin width; disjoint object and surface; pointwise predictions continuous'))
if __name__=='__main__':main()
