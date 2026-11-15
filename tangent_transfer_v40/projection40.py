"""Paired geometry control, evaluated on a preselected half of the panels' blocks."""
import json
import os
import time
from pathlib import Path
os.environ['BELIEF_DEVICE']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import run40
from scipy.stats import t as student

root=Path(__file__).resolve().parent
v38=run40.previous
world=run40.world


def decode(inner, factual, observed, direction):
    sigma=inner.sigma[v38.Q]
    u=direction/sigma
    u=u/u.norm()
    bank=inner.white[v38.Q]
    orth=bank-(bank@u)[:,None]*u
    norm=.5*orth.square().sum(-1)
    y=v38.t(observed)/sigma
    prior=inner.fit(v38.t(factual)).clamp_min(1.e-300).log()[:,None,:]
    posterior=v38.torch.softmax(prior + y@orth.T - norm, dim=-1)
    return (posterior@inner.label[:,0]).numpy()/np.sqrt(2.)


def evaluate(panel,block,family,law,direction):
    c=run40.CELLS.index(f'{family}_base_{law}')
    path=run40.block_path(c,panel,block)
    meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as x:
        base=x['p_base'].copy()
        alpha=x['alpha'];actions=x['actions'];labels=x['labels'];z=x['z'];w=x['weights']
    max_shift_error=0.
    for e,a in enumerate(alpha):
        inner=v38.Engine(meta['pair'],*a).inner
        for si,sign in enumerate(run40.SIGNS):
            context=(family,'base',law,sign,panel)
            factual,target,_,obs=world.deployment(meta['deployment_seeds'][e],a,context,'strong',32)
            np.testing.assert_array_equal(target,labels[si,0,e])
            mask=actions[si,e]==v38.Q
            indices=np.flatnonzero(mask)
            for j in range(0,len(indices),16):
                selected=indices[j:j+16]
                y=obs[v38.Q][selected]
                px=decode(inner,factual[selected],y,direction)
                if j==0:
                    for signed in (-1,1):
                        moved=decode(inner,factual[selected],y+signed*direction.numpy(),direction)
                        err=np.max(np.abs(moved-px));max_shift_error=max(max_shift_error,err)
                        if err>2e-10:
                            raise ValueError('Non-invariant query update')
                base[si,e,selected]=px
    def risk(multiplier):
        shift=z[:,None,:,None,None]*multiplier
        p=v38.apply(base[:,None],shift,v38.BETA)
        return np.square(p-labels[...,None]).mean()
    return np.array([risk(.325),risk(w[:,:,:,None,None])]),max_shift_error


def bounds(data):
    m=data.mean(0)
    rad=student.ppf(.975,11)*data.std(0,ddof=1)/np.sqrt(12)
    return {'mean':m.tolist(),'lo':(m-rad).tolist(),'hi':(m+rad).tolist()}


def main():
    v38.torch.set_num_threads(1)
    direction=world.perturbation(world.VYS[0],'shelf','bulge')[1]-world.perturbation(world.VYS[0],'shelf','base')[1]
    checkpoint=root/'projection_results'
    checkpoint.mkdir(exist_ok=True)
    for panel in range(12):
        block=panel%2
        for fam in world.FAMILIES:
            for law in world.SURFACES:
                p=checkpoint/f'{panel:02d}_{fam}_{law}.npz'
                if p.exists():
                    continue
                cells=[run40.CELLS.index(f'{fam}_{v}_{law}') for v in world.CONTOURS]
                paths=[run40.block_path(c,panel,block) for c in cells]
                while not all(q.with_suffix('.json').exists() for q in paths):
                    time.sleep(5)
                projected,err=evaluate(panel,block,fam,law,direction)
                raw=[]
                for q in paths:
                    with np.load(q) as obj:
                        raw.append(obj['risk'].mean((0,1,3)))
                v38.atomic_npz(p,original=raw,projected=projected,err=np.array(err))
        print('completed projection panel',panel,flush=True)
    orig=np.empty((12,2,2,3,2))
    projected=np.empty((12,2,2,2))
    err=0.
    for i in range(12):
        for f,fam in enumerate(world.FAMILIES):
            for k,law in enumerate(world.SURFACES):
                with np.load(checkpoint/f'{i:02d}_{fam}_{law}.npz') as u:
                    orig[i,f,k]=u['original']; projected[i,f,k]=u['projected'];err=max(err,float(u['err']))
    orig_panel=orig.mean((1,2,3))
    proj_panel=projected.mean((1,2))
    result={'original':orig_panel.mean(0).tolist(),'projected':proj_panel.mean(0).tolist(),
            'projected_minus_original':bounds(proj_panel-orig_panel),
            'projected_optimizer_minus_fixed':bounds(proj_panel[:,1]-proj_panel[:,0]),
            'per_variant_change':{v:bounds(proj_panel-orig[:,:,:,j].mean((1,2))) for j,v in enumerate(world.CONTOURS)},
            'per_family_law':projected.mean(0).tolist(),'max_invariance_error_checked':err,'blocks':48}
    v38.atomic_json(root/'projection40.json',result)


if __name__=='__main__':
    main()
