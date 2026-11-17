"""Privileged pre-acquisition forecast using independent fourth-pair donor records.

Estimates latent conditional purchase value at locked held prefixes; in contrast,
a hindsight risk-minimizer looks at both factual outcomes for the current fourth object.
"""
import json
from pathlib import Path
import os
import sys
import numpy as np
from score54 import ROOT,V53,V52,IMPACT,REFERENCE,N,R,GM,mix_event,quantiles,pair_ci
from policy53 import witnesses53, DRIFT_PRIOR, likelihood_grid, direction_likelihood, RETENTIONS, PRIORS, ERROR_MOMENTS, V50, bank


def posterior_moments(ow,of,nw,nf):
    k=RETENTIONS
    old=likelihood_grid(ow,of,'likelihood')
    new=likelihood_grid(nw,nf,'likelihood')
    mass=new*((1-DRIFT_PRIOR)*old+DRIFT_PRIOR*old[...,::-1])*PRIORS
    mass/=mass.sum(-1,keepdims=True)
    mu,m2=ERROR_MOMENTS['nominal']; mean=mass@k; second=mass@(k*k)
    q=mu+(1-2*mu)*mean
    q2=second+2*mu*(mean-2*second)+m2*(1-4*mean+4*second)
    qk=mu*mean+(1-2*mu)*second
    return mean,q,q2,qk


def control(panel):
    file=f'panel_{panel:02d}.npz'
    rng=np.random.default_rng([540911,panel])
    order=bank(panel)[2]
    with np.load(V50/'cal50'/file) as f:
        ids=f['cal_object_id'][:,order]
    env=np.arange(4)[:,None,None]
    valid=(ids[:,None,:,3,None]!=ids[:,:,:3,None].transpose(0,1,3,2)).all(-1)
    # Current prefix excludes its own future and incompatible previously paid specimens.
    valid &= ~np.eye(R,dtype=bool)[None]
    assert valid.sum(-1).min()>128
    donors=np.argsort(np.where(valid,rng.random(valid.shape),2.),axis=-1)[...,:128]
    expected=np.empty((3,2,5,4,R,4))
    realized=np.empty_like(expected,dtype=np.float32)
    rate=np.empty_like(expected)
    with np.load(V52/'risk52'/file) as f:
        ex=f['exact'][0]; raw=f['realized']
    with np.load(V53/'decisions53'/file) as f:
        methods=list(f['methods'].astype(str))
        a=f['alpha'][...,[methods.index(m) for m in ('reference4_joint','impact_joint')]]
    for e,ev,k in np.ndindex(3,2,5):
        ow,of,nw,nf,s,_=witnesses53(panel,k,4-k if ev else k,e)
        mk,q,q2,qk=[v[...,None] for v in posterior_moments(ow,of,nw[...,:3],nf[...,:3])]
        # L_ref = c + d*k, since only one new observation is considered.
        L=likelihood_grid(nw[...,3:4],nf[...,3:4],'likelihood')
        dr=(L[...,-1]-L[...,0])/(RETENTIONS[-1]-RETENTIONS[0])
        cr=L[...,0]-dr*RETENTIONS[0]
        ci,di=direction_likelihood(nw[...,3],s,'likelihood')
        on_self=np.stack([((cr[...,None]*q+dr[...,None]*qk)/(cr[...,None]+dr[...,None]*mk))[...,0],
                          ((ci[...,None]*q+di[...,None]*q2)/(ci[...,None]+di[...,None]*q))[...,0]],-1)
        ix=e,ev,k
        np.testing.assert_allclose(on_self,a[ix],atol=1e-11,rtol=0)
        cr,dr,ci,di=[v[env,donors] for v in (cr,dr,ci,di)]
        ar=(cr*q+dr*qk)/(cr+dr*mk);ai=(ci*q+di*q2)/(ci+di*q)
        p,m,_,gap=ex[ix].T
        def brier(alpha,p,m,gap):
            return m+(p-m)*alpha-gap*alpha*(1-alpha)
        delta=brier(ai,p[:,None,None],m[:,None,None],gap[:,None,None])-brier(ar,p[:,None,None],m[:,None,None],gap[:,None,None])
        votes=[delta[...,:n].mean(-1)<0 for n in (8,32,128)]
        # Hindsight operates on actual potential observations on this witness.
        fixed=brier(on_self,p[:,None,None],m[:,None,None],gap[:,None,None])
        votes.append(fixed[...,1]<fixed[...,0])
        ra=np.stack(votes,-1); rate[ix]=ra
        al=np.where(ra,a[ix][...,1,None],a[ix][...,0,None])
        expected[ix]=brier(al,p[:,None,None],m[:,None,None],gap[:,None,None])
        p,m,_,gap=np.moveaxis(raw[ix],-1,0)
        realized[ix]=brier(al,p[...,None],m[...,None],gap[...,None])
    return expected,realized,rate

def main():
    assert os.getenv('SLURM_JOB_ID')
    blocks=[control(i) for i in range(N)]
    exact,real,route=[np.stack(x) for x in zip(*blocks)]
    means=mix_event(exact).mean((1,2,3,4));rates=mix_event(route).mean((1,2,3,4));q=quantiles(real,GM)
    d=json.loads((ROOT/'score54.json').read_text()); names=d['methods']
    with np.load(ROOT/'catalog54.npz') as f: orig=f['means'][:,0]
    comps={n:[pair_ci(means[:,i]-orig[:,names.index(n)],1).tolist() for i in range(4)] for n in ('adaptive','fixed4R','fixed3RI','latent_cell_loo','both_futures_min_mean')}
    np.testing.assert_allclose(means[:,-1],orig[:,names.index('both_futures_min_mean')],rtol=0,atol=5e-12)
    out=dict(draws=[8,32,128,'hindsight'], mean=means.mean(0).tolist(),p90=q.tolist(),impact_rate=rates.mean(0).tolist(), delta=comps)
    (ROOT/'prospective54.json').write_text(json.dumps(out,indent=1))
    print(out['mean'],q)
if __name__=='__main__':main()
