"""Store decisions based only on the paid prefix, before importing any deployment outcome."""
import json
import os
from pathlib import Path
import numpy as np
from policy53 import (
    ROOT, V52, METHODS, EPS, RETENTIONS, PRIORS, DRIFT_PRIOR, bank,
    witnesses53, mixed53, choice53, load_model, direction_likelihood, likelihood_grid,
)


def check_quadrature(raw,model):
    ow,of,nw,nf,sign,noisy = (v[0,:8] for v in raw)
    # Independent numeric integration of the nonuniform e density and joint posterior.
    k = RETENTIONS[:,None]
    node,w = np.polynomial.legendre.leggauss(24)
    e = (node+1)*.25
    ew = .25*w*8*(1-2*e)**3
    np.testing.assert_allclose(ew.sum(),1,atol=1e-13)
    q = k+(1-2*k)*e
    lo = likelihood_grid(ow,of,'likelihood')
    ln = likelihood_grid(nw[...,:3],nf[...,:3],'likelihood')
    a,d = direction_likelihood(nw[...,3],sign,'likelihood')
    mass = (a[:,None,None]+d[:,None,None]*q)*ew*PRIORS[:,None]
    mass *= (ln*((1-DRIFT_PRIOR)*lo+DRIFT_PRIOR*lo[...,::-1]))[...,None]
    target = (mass*q).sum((1,2))/mass.sum((1,2))
    val = choice53('impact_joint',ow,of,nw,nf,sign,noisy,model)[1]
    np.testing.assert_allclose(val,target,atol=5e-14)
    # A changed missing (fourth reference) value has zero effect in every mixed arm.
    poison = nf.copy()
    poison[...,3]=np.nan
    for method in METHODS:
        if method.startswith('reference4'):
            continue
        for i,j in zip(choice53(method,ow,of,nw,nf,sign,noisy,model),
                       choice53(method,ow,of,nw,poison,sign,noisy,model)):
            np.testing.assert_array_equal(i,j)


def panel(i,model):
    name = f'panel_{i:02d}.npz'
    shape = (3,2,5,4,256)
    alpha = np.empty((*shape,len(METHODS)),dtype=np.float64)
    action = np.empty(alpha.shape,dtype=np.int8)
    error = np.empty(shape)
    posterior_k = np.empty(shape)
    weak_agree = np.empty(shape, np.int8)
    bits = np.empty(shape, np.int8)
    noisybits = np.empty(shape, np.int8)
    weak_proxy = np.empty(shape,np.float32)
    with np.load(V52/'decisions52'/name) as f:
        ref4 = f['target_alpha'][:,:,3,...,0]
    for ei,ev,ki in np.ndindex(3,2,5):
        knew = 4-ki if ev else ki
        ix = ei,ev,ki
        raw = witnesses53(i,ki,knew,ei)
        ow,of,nw,nf,s,noise=raw
        bits[ix]=s
        noisybits[ix]=noise
        weak_proxy[ix]=nw[...,3]
        weak_agree[ix]=np.where(s!=0,np.sign(nw[...,3])==s,-1)
        d=mixed53(ow,of,nw[...,:3],nf[...,:3],nw[...,3],s)
        error[ix],posterior_k[ix] = d['post_e'],d['post_k']
        for m,method in enumerate(METHODS):
            act,mix=choice53(method,*raw,model)
            alpha[ix+(Ellipsis,m)] = mix
            action[ix+(Ellipsis,m)] = act
        np.testing.assert_allclose(alpha[ix+(Ellipsis,METHODS.index('reference4_joint'))],ref4[ki,knew],atol=2.e-12)
        masked=alpha[ix+(Ellipsis,METHODS.index('impact_joint_masked'))]
        mixed=alpha[ix+(Ellipsis,0)]
        np.testing.assert_allclose(masked[s==0],mixed[s==0],atol=1e-14)
        if ix==(0,1,0):
            check_quadrature(raw,model)
    np.testing.assert_array_equal(alpha[:,0,2],alpha[:,1,2])
    assert np.all((alpha>=0)&(alpha<=1))
    dest=ROOT/'decisions53'/name
    dest.parent.mkdir(exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,alpha=alpha, unsigned=action==2, action=action,methods=METHODS,
                        post_e=error,post_k=posterior_k,impact_sign=bits,impact_noisy=noisybits,
                        observed_weak4=weak_proxy,weak_sign_agree=weak_agree)
    tmp.replace(dest)
    print('witness cache',i,flush=True)


def summary():
    data={k:[] for k in ['post_e','weak_sign_agree','nonzero','read_error','q_abs_change','q_by_e','q_by_e_masked']}
    for i in range(12):
        with np.load(ROOT/'decisions53'/f'panel_{i:02d}.npz') as f:
            q=f['alpha'][...,0];m=f['alpha'][...,METHODS.index('impact_joint_masked')]
            s=f['impact_sign'];active=s!=0
            # Conditional diagnostic weighting: average four prespecified g values, then catalogs.
            w=np.array([.5625,.4375])
            pool=lambda a: np.einsum('ij,j->i',a.mean(axis=(2,3,4)),w)
            for key,v in [('post_e',pool(f['post_e'])),('q_by_e',pool(q)),('q_by_e_masked',pool(m)),
                          ('nonzero',pool(active)),('read_error',pool(f['impact_noisy']!=s)/pool(active)),
                          ('q_abs_change',pool(abs(q-m))),
                          ('weak_sign_agree',pool(f['weak_sign_agree']==1)/pool(active))]:
                data[key].append(v)
    out={k:np.mean(v,axis=0).tolist() for k,v in data.items()}
    out['epsilon']=EPS
    out['scope']='Twelve catalog means; equal five pre-retention strata and 4 drift mixtures. No deployment outcomes.'
    (ROOT/'calibration53.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    coefficients=load_model()
    for cat in range(12):
        panel(cat,coefficients)
    summary()
