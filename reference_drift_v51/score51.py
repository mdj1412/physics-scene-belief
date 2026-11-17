"""Paired exact Brier risks under specimen-level and session-level sign changes."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
V50=ROOT.parent/'polarity_calibration_v50'
sys.path.insert(0,str(V50))
from estimator50 import DEAD_ZONE,infer
from score50 import interval,blend
from world51 import paired_calibration_rows
GRID=np.array([0.,.1,.25,.5,.75,1.])
K=np.array([0.,.25,.5,.75,1.])
PRIOR=np.array([.5,.25,.25])
METHODS=['likelihood_soft','vote_soft','unsigned_fit','vote_hard','unsigned_symmetric','known_rates_soft','known_rates_vote','known_drift_event_soft']


def conditional_risks(path,forecasts):
    with np.load(path) as f:
        y=torch.from_numpy(f['labels'].astype(float)).cuda()
    probs=torch.from_numpy(forecasts).cuda() # M,F,H,S,A,V,E,O,R
    pmean=probs.mean(dim=(4,5,8))
    ymean=y.mean(dim=(3,4))
    p2=(probs*probs).mean(dim=(1,2,4,5,7,8)).transpose(1,2)
    cross=torch.einsum('mfhseo,fhteo->mest',1-2*pmean,ymean)/(4*2*128)
    kernel=(p2[...,None]+cross).cpu().numpy() # M,E,S,T
    gap=(probs[0]-probs[1]).square().mean(dim=(0,1,3,4,6,7)).cpu().numpy() # S,E
    gap=PRIOR@gap
    # Direct squared error over each of the six binary radius/offset outcomes, alternating signed states.
    m,s,t=0,1,2
    observed=probs[m,:,:,s]
    checked=sum((observed-y[:,:,t,r,x][:,:,None,None,:,:,None]).square().mean(dim=(0,1,2,3,5,6)) for r,x in np.ndindex(2,3))/6
    np.testing.assert_allclose(checked.cpu().numpy(),kernel[m,:,s,t],atol=1e-12,rtol=0)
    plus=np.einsum('mest,st->me',kernel,np.diag(PRIOR))
    reverse=np.einsum('mest,st->me',kernel,np.diag(PRIOR)[:,[0,2,1]])
    return plus,reverse,gap


def weights(cal,permutation):
    with np.load(cal) as f:
        w=f['weak_mm'][:,permutation,:,:4]; z=f['full_mm'][:,permutation,:,:4]
        a=infer(w,z)
        np.testing.assert_allclose(a,f['khat'][:,2,permutation,:],atol=1e-12,rtol=0)
    readable=(abs(w)>=DEAD_ZONE[0])&(abs(z)>=DEAD_ZONE[1])
    counts=(readable&((w>0)==(z>0))).sum(-1)
    b=(1.+counts)/(2.+readable.sum(-1))
    return a,b


def panel(ends,cal):
    keep,flip,gap=ends
    a,b=cal
    gshape=(len(GRID),len(GRID),5,4,len(METHODS))
    values=np.empty(gshape)
    for i,e in enumerate(GRID):
        for j,g in enumerate(GRID):
            eff=e+g-2*e*g
            retention=eff+(1-2*eff)*K
            def endpoints(ret):
                return flip[:,None]+(keep-flip)[:,None]*ret[None,:,None]
            lp,lm,uns=endpoints(retention)
            def mix(coef,pr=lp,mr=lm):
                return blend(pr,mr,gap,coef.mean(-1),np.square(coef).mean(-1))
            v=values[i,j]
            v[...,0]=mix(a)
            v[...,1]=mix(b)
            v[...,2]=uns
            v[...,3]=mix((b>=.5).astype(float))
            v[...,4]=blend(lp,lm,gap,.5,.25)
            v[...,5]=mix(eff+(1-2*eff)*a)
            v[...,6]=mix(eff+(1-2*eff)*b)
            def if_event(inverted):
                rate=(1-e) if inverted else e
                l,m,_=endpoints(rate+(1-2*rate)*K)
                return mix(rate+(1-2*rate)*a,l,m)
            v[...,7]=(1-g)*if_event(False)+g*if_event(True)
    return values


def summarize(x,tag):
    # x catalog,e,g,pre-kappa,env,method
    loss=x.mean((3,4)); contrast=loss[...,0,None]-loss[...,[1,2]]
    ci=interval(contrast,72)
    roots=[]
    rng=np.random.default_rng(5114221)
    idx=rng.integers(x.shape[0],size=(20000,x.shape[0]))
    for m in (0,1):
        d=contrast[:,:,0,m]
        def root(a):return a[...,0]/(a[...,0]-a[...,-1])
        roots.append([root(d.mean(0)),*np.quantile(root(d[idx].mean(1)),[.025,.975])])
    attenuation=(1-2*GRID[:,None])*(1-2*GRID)
    for m in range(5):
        estimated=(loss[:,0,0,m,None,None]+loss[:,-1,0,m,None,None])/2+(loss[:,0,0,m,None,None]-loss[:,-1,0,m,None,None])*attenuation/2
        np.testing.assert_allclose(estimated,loss[...,m],atol=2e-12,rtol=0)
    np.savez_compressed(ROOT/f'catalog51{tag}.npz',risk=x,e=GRID,g=GRID,pre_kappa=K,methods=METHODS)
    out=dict(methods=METHODS,e=GRID,g=GRID,pre_kappa=K,loss=loss.mean(0),effect_simultaneous95=ci,
             n_catalogs_lower=(contrast<0).sum(0),by_pre_kappa=x.mean((0,4)),root_equivalent_flip_pointwise95=roots,
             both_upper_below_zero=(ci[...,2]<0).all(-1),
             scope='12 paired catalogs; uniform 5 calibration retentions. Both interventions independent. Frozen deployment policies blind to both rates.')
    (ROOT/f'score51{tag}.json').write_text(json.dumps(out,default=lambda t:np.asarray(t).tolist(),indent=1))
    return out


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available()
    results=[[],[]]; no_replace=[]
    for i in range(12):
        name=f'panel_{i:02d}.npz'
        with np.load(V50/'forecast50'/name) as f:
            inds=[list(f['methods'].astype(str)).index(m) for m in ('local_plus','local_minus','local_unsigned_fit')]
            p=f['p'][inds]
        with np.load(V50/'worlds'/name) as f:
            partners=paired_calibration_rows(f['surface'],f['law'])
        assert list(partners)==[2,3,0,1]
        ca=V50/'cal50'/name
        a=weights(ca,partners)
        for level,path in enumerate(('labels50','labels50_ultra')):
            ends=conditional_risks(V50/path/name,p)
            results[level].append(panel(ends,a))
            if level==0:
                no_replace.append(panel(ends,weights(ca,np.arange(4)))[0,0].mean((0,1)))
    old=json.loads((V50/'controls50.json').read_text())['uniform'][2]
    mirror=np.mean(no_replace,0)
    np.testing.assert_allclose(mirror[:2],old[2:4],rtol=0,atol=2e-12)
    np.testing.assert_allclose(mirror[3],old[1],rtol=0,atol=2e-12)
    old_unsigned=json.loads((V50/'score50.json').read_text())['uniform_mean'][2][4]
    np.testing.assert_allclose(mirror[2],old_unsigned,rtol=0,atol=2e-12)
    for data,tag in zip(results,('','_ultra')):
        out=summarize(np.stack(data),tag)
        print(tag,'double wins',np.sum(out['both_upper_below_zero']),'roots',out['root_equivalent_flip_pointwise95'],flush=True)
    (ROOT/'replay51.json').write_text(json.dumps(dict(same_surface_null_replay=mirror.tolist(),gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__),allocation=os.getenv('SLURM_JOB_ID'),paired_surfaces=[2,3,0,1])))
if __name__=='__main__':
    with torch.inference_mode():main()
