"""Supplemental post-calibration sensor selector. Truth and alternative observations only in fit/score."""
import sys,json
from pathlib import Path
import numpy as np
from decisions34 import ROOT,OUT,band,sample,dump
from direct_head34 import FAMILIES,SCHEDULES,cue_est
from target34 import deployment_x
def inputs(x,g):
    assert x.shape==(8,128,11) and g.shape==(8,128)
    # Pre-observation rank inside deployment quota, with stable slot ordering.
    rank=np.argsort(np.argsort(g,axis=1,kind='stable'),axis=1,kind='stable')/127
    v=np.stack([g,rank,x[:,:,:9].mean(2),x[:,:,8]-x[:,:,0],x[:,:,0]+x[:,:,8]-2*x[:,:,4],x[:,:,9],x[:,:,10],g*x[:,:,9]],-1)
    return v
def run():
    lk=json.loads((ROOT/'diag_models.json').read_text());v=[];cues=[];cost=[];orig=[]
    for f,fam in enumerate(FAMILIES):
        vr=[];co=[];cu=[];oo=[]
        for b in range(8):
            base=901000000+f*1000000+b*10000
            for d,sl in enumerate(('plus','minus')):
                law=f'{fam}_{sl}'
                for a,sc in enumerate(SCHEDULES):
                    p=OUT/law/f'block_{b:03d}'/sc
                    with np.load(p/'choices.npz') as z,np.load(p/'metrics.npz') as m:
                        if d==0 and a==0:
                            xx=np.array([deployment_x(base+200+i,alpha,law) for i,alpha in enumerate(z['alpha'])])
                        vr.append(inputs(xx,z['gaps'][:,1]))
                        cu.append([cue_est(lk,fold,z['paid_x'][:,:4],z['paid_y'][:,:4],z['paid_direction'][:,:4],sc) for fold in (f,3-f)])
                        losses=((m['probabilities']-m['labels'][:,:,None,None])**2).mean(3)
                        choice=m['actions'][:,:,1]==3
                        assert np.allclose(np.where(choice,losses[:,:,1],losses[:,:,0]).mean(1),m['brier'][:,1])
                        co.append(losses);oo.append(choice)
        v.append(np.array(vr).reshape(8,2,2,8,128,-1))
        cost.append(np.array(co).reshape(8,2,2,8,128,2));cues.append(np.array(cu).reshape(8,2,2,2,8));orig.append(np.array(oo).reshape(8,2,2,8,128))
    c=np.array(cost);oc=np.array(orig);cu=np.array(cues);x=np.array(v)
    rng=np.random.default_rng(3409122);flip=np.zeros((4,8),dtype=int)
    for f in range(4):
        for r in (0,1):flip[f,2*rng.permutation(4)[:2]+r]=1
    def mats(f,fold,mode):
        z=cu[f,...,0 if f==fold else 1,:]
        if mode==0:z=z*0
        if mode==2:z=np.broadcast_to(np.array([1,-1])[None,:,None,None],z.shape)
        if mode==3:z=np.stack([u[::-1] if flip[f,i] else u for i,u in enumerate(z)])
        # Returning N x features would include (unneeded) orientation intercept; retain it for matched heads.
        ca=np.broadcast_to(z[...,None,None],x[f].shape[:-1]+(1,))
        return np.concatenate((x[f],ca,ca*x[f]),-1)
    vals=np.zeros((4,8,2,2,5)); changes=vals.copy()
    for f in range(4):
        g=3-f
        vals[f,...,0]=np.where(oc[f],c[f,...,1],c[f,...,0]).mean((-1,-2))
        for j in range(4):
            tr=mats(g,f,j);te=mats(f,f,j); a=tr.reshape(-1,tr.shape[-1])
            y=(c[g,...,0]-c[g,...,1]).flatten();mu=a.mean(0);sd=a.std(0);sd[sd<1e-10]=1
            a=(a-mu)/sd; beta=np.linalg.solve(a.T@a/len(a)+.1*np.eye(a.shape[-1]),a.T@(y-y.mean())/len(a))
            pred=((te-mu)/sd)@beta+y.mean();ids=np.argsort(-pred,axis=-1,kind='stable')[...,:13]
            selected=np.zeros_like(oc[f]);np.put_along_axis(selected,ids,True,axis=-1)
            assert np.all(selected.sum(-1)==13)
            vals[f,...,j+1]=np.where(selected,c[f,...,1],c[f,...,0]).mean((-1,-2))
            changes[f,...,j+1]=(selected!=oc[f]).mean((-1,-2))
    mu=vals.mean((0,1,2))
    def dif(v):return np.stack([v[...,a,i]-v[...,a,j] for a in (0,1) for i,j in ((2,1),(3,1),(2,4),(2,0))],-1)
    idx=sample(np.random.default_rng(340794),12000)
    dd=dif(mu);draw=dif(vals[np.arange(4)[None,:,None],idx].mean((1,2,3)))
    dump('selector_stats34.json',dict(methods=['original','neutral','estimated','oracle','placebo'],means=mu,delta=dd,ci=band(dd,draw),percent_changed_vs_original=100*changes.mean((0,1,2))))
if __name__=='__main__':run()
