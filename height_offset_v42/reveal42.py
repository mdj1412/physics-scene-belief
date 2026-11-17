"""Reveal only physical query height, using matched nominal sensor likelihoods."""
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import t
import runner42 as job
import stats42 as s
r=job.r
ROOT=job.ROOT

def predict(bank,template,x,actions,sensors,event_matrix):
    white,norm,_=template
    prior=bank.fit(r.v38.t(x))
    out=np.empty((len(x),len(s.OFF)))
    for a in (r.Q,r.P):
        ids=np.flatnonzero(actions==a)
        if len(ids)==0: continue
        y=r.v38.t(sensors[a][ids,0])
        if a==r.Q:
            ll=(y/bank.sigma[a])@white.T-norm
            post=r.torch.softmax(prior[ids].clamp_min(1e-300).log()+ll,-1)
        else:
            post=bank.update(prior[ids],a,y)
        out[ids]=(post@event_matrix).cpu().numpy()
    return out

def panel(n):
    paths=[s.raw(row,n) for row in s.CELLS]
    while not all(p.with_suffix('.json').exists() for p in paths): time.sleep(6)
    cache=ROOT/'reveal_outputs'/f'panel_{n:02d}.json'
    info=json.loads((ROOT/'results/run.json').read_text()); digest=r.v38.digest(Path(__file__))
    if cache.exists():
        last=json.loads(cache.read_text())
        if last['analysis']==digest and last['fingerprint']==info['fingerprint']: return last
    meta=json.loads(paths[0].with_suffix('.json').read_text())
    with np.load(paths[0]) as a:alpha=a['alpha']
    banks=[r.v38.Engine(meta['model_bank_seed'],*u).inner for u in alpha]
    matrices={};templates={};rows=[];peak=0.
    for f,hk in ((f,h) for f in s.FAMILY for h in s.HEIGHT):
        h=r.hvalue(hk)
        with np.load(job.OUT/'directions'/f'panel_{n:02d}_{f}_{hk}.npz') as d:
            templates[f,hk]=[r.templates(banks,h,u) for u in (None,d['oracle'],d['estimated'])]
        if hk not in matrices:
            matrices[hk]=[r.torch.stack([r.impact(b.theta,h+j*.001,r.QUERY_INCOMING)[1].to(b.theta.dtype)
                                         for j in s.OFF],dim=1) for b in banks]
    for row,path in zip(s.CELLS,paths):
        fam,hkey,var,law=row; h=r.hvalue(hkey)
        with np.load(path) as a:
            choices=a['actions'][:,:,:32]; observed=a['p_base'][:,:,:,:32,0]; target=a['labels'][:,:,:,:,:32]
        probs=np.zeros((3,2,8,32,9))
        for si,sign in enumerate(r.SIGNS):
            ctx=(fam,var,h,law,sign,job.w.catalog_panel(n))
            for e,(bank,seed,aa) in enumerate(zip(banks,meta['deployment_seeds'],alpha)):
                x,y,_,sensors=r.world.deployment(seed,aa,ctx,r.REGIMES[0],job.REPEATS)
                np.testing.assert_array_equal(target[s.Z,si,0,e],y[:32])
                for di in range(3):
                    probs[di,si,e]=predict(bank,templates[fam,hkey][di][e],x[:32],choices[si,e],sensors,matrices[hkey][e])
        err=np.abs(probs[...,s.Z]-observed).max()
        if err>2e-10: raise ValueError(f'Nominal CPU/GPU mismatch: {err}')
        peak=max(peak,err)
        actual=target.transpose(0,1,2,3,4)[:,None]
        changed=probs.transpose(4,0,1,2,3)[:,:,:,None]
        baseline=observed[None,:,:,None,:,:]
        rows.append(np.stack([np.square(baseline-actual).mean((2,3,4,5)),
                              np.square(changed-actual).mean((2,3,4,5))]).tolist())
    data=dict(risk=rows,peak=peak,fingerprint=info['fingerprint'],analysis=digest)
    cache.parent.mkdir(exist_ok=True)
    r.v38.atomic_json(cache,data)
    print('CPU readout complete',n,flush=True)
    return data

def main():
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    r.torch.set_num_threads(1)
    with r.torch.inference_mode(): records=[panel(i) for i in range(12)]
    # panel x cell x hidden/revealed x offset x decoder
    arr=np.array([v['risk'] for v in records])
    v=arr.mean(1); gap=v[...,2]-v[...,0]
    groups=arr.reshape((12,2,2,3,2,2,9,3)).mean((3,4))
    out=dict(n=12,objects_per_env=32,draw=0,mean=v.mean(0),paired_projection_gap=s.ci(gap,9),
             revealm_hidden=s.ci(v[:,1]-v[:,0],9),
             group_projection_gap=s.ci(groups[...,2]-groups[...,0],36),
             max_nominal_reproduction_error=max(x['peak'] for x in records))
    (ROOT/'reveal42.json').write_text(json.dumps(out,indent=2,default=lambda v:v.tolist(),allow_nan=False)+'\n')
if __name__=='__main__':main()
