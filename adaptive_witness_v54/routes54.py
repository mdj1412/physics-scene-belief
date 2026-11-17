"""Blind routing before importing any target labels, risks or potential fourth readings."""
import os
from pathlib import Path
import json
import numpy as np
from policy54 import (ROOT, MODES, V53, select54, values54, tables, channels,
                      RETENTIONS, PRIORS, likelihood_grid, ERROR_MOMENTS)
from policy52 import V50, bank


def raw_prefix(panel):
    # Only the permitted prefix is returned to callers. The permutation is the
    # public v52/53 replay convention for environment rows, not an input feature.
    with np.load(V50/'cal50'/f'panel_{panel:02d}.npz') as f:
        w,r = f['weak_mm'][...,:3],f['full_mm'][...,:3]
    order = np.random.default_rng(520096+panel).permutation(w.shape[-2])
    return w[:,:,order], r[:,:,order]


def exact_check(weak,ref,mode):
    """Independent direct sum over future sensor bins and historical observations."""
    _,_,_,hist,_,_,_,coeff=tables()
    prior=likelihood_grid(weak,ref,'likelihood')*PRIORS
    prior/=prior.sum()
    unorm=hist*prior
    ph=unorm.sum(1)
    pk=unorm/ph[:,None]
    k=RETENTIONS
    m,s2=ERROR_MOMENTS['nominal']
    mt,st=ERROR_MOMENTS['uniform' if mode=='robust' else 'nominal']
    moments=lambda m,s2: (k+(1-2*k)*m, k*k+2*k*(1-2*k)*m+(1-2*k)**2*s2)
    qn,qn2=moments(m,s2);qt,qt2=moments(mt,st)
    def score(q,alpha):
        c=coeff[0,1]+q*(coeff[1,1]-coeff[0,1])
        return alpha*(c+coeff[0,2]*alpha)
    base=ph@score(pk@qt,pk@qn)
    values=[]
    for n,c in enumerate(channels(96)):
        mask=c.sum(axis=0)>1e-16
        a,d=c[0,mask],(c[1]-c[0])[mask]
        tn,tt=(k,k) if n==0 else (qn,qt)
        qtn,qtt=(qn*k,qt*k) if n==0 else (qn2,qt2)
        pn=a[None,:]+(pk@tn)[:,None]*d; pt=a[None,:]+(pk@tt)[:,None]*d
        qnom=((pk@qn)[:,None]*a+(pk@qtn)[:,None]*d)/pn
        qtrue=((pk@qt)[:,None]*a+(pk@qtt)[:,None]*d)/pt
        values.append(base-ph@(pt*score(qtrue,qnom)).sum(1))
    np.testing.assert_allclose(values,values54(weak,ref,mode),atol=1e-9,rtol=1e-6)


def main():
    assert os.getenv('SLURM_JOB_ID')
    dest=ROOT/'routes54'; dest.mkdir(exist_ok=True)
    counts=[]; differences=[]; extrema=[]
    for panel in range(12):
        w,r=raw_prefix(panel)
        assert w.shape==r.shape==(5,4,256,3)
        arrays={};fine=[];ext=[]
        for mode in MODES:
            key='impact' if mode=='evsi' else mode
            rg,ig=values54(w,r,mode)
            route=ig>rg
            arrays[key]=route
            arrays['delta' if mode=='evsi' else mode+'_delta']=ig-rg
            ext.append([float((ig-rg).min()),float((ig-rg).max())])
            ix=(slice(None),slice(None),slice(0,256,17))
            rr,ii=values54(w[ix],r[ix],mode,subdivisions=160,fine_old=True)
            fine.append([int(np.sum((ii>rr)!=route[ix])),int(ii.size),float(np.max(np.abs((ii-rr)-(ig-rg)[ix])))])
        np.savez_compressed(dest/f'panel_{panel:02d}.npz',**arrays)
        counts.append([arrays['impact' if m=='evsi' else m].mean((1,2)).tolist() for m in MODES])
        differences.append(fine);extrema.append(ext)
        if panel==0:
            for m in ('evsi','robust'):
                for i in range(3): exact_check(w[i,0,i],r[i,0,i],m)
            try:
                select54(np.ones(4),np.ones(4))
            except ValueError: pass
            else: raise RuntimeError('fourth measurement accepted by selection')
    d=np.array(differences)
    js=dict(modes=MODES,route_rate_by_new_k=np.mean(counts,0).tolist(),rate=np.mean(counts,(0,2)).tolist(),
            fine_disagreements=d[:,:,0].astype(int).sum(0).tolist(),fine_n=d[:,:,1].astype(int).sum(0).tolist(),
            max_delta_error=d[:,:,2].max(0).tolist(),
            delta_range=np.stack([np.array(extrema)[:,:,0].min(0),np.array(extrema)[:,:,1].max(0)],axis=-1).tolist())
    (ROOT/'routes54.json').write_text(json.dumps(js,indent=2))
    print(js,flush=True)


if __name__=='__main__':
    main()
