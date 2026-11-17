"""Matched supplementary comparison with finite-variance Gaussian nuisance marginalization."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import runner41 as r
import sham41

ROOT=r.ROOT

def items(banks,height,pairs):
    unit=r.world.estimate_direction(pairs)
    delta=(pairs[:,1]-pairs[:,0])/.00030
    # The paired scans have independent amplitudes and independent sensor errors of unit variance.
    variance=max(0.,float(np.var(delta@unit,ddof=1)/2-1))
    shrink=variance/(1+variance)
    u=r.v38.t(unit)
    out=[]
    for white,_,label in r.templates(banks,height):
        precision_mean=white-shrink*(white@u)[:,None]*u
        out.append((precision_mean,.5*(white*precision_mean).sum(-1),label))
    return out,variance

def panel_rows(panel,wait):
    b=panel%2
    paths=[r.rawpath(i,panel,b) for i in range(len(r.CELLS))]
    while not all(p.with_suffix('.json').exists() for p in paths):
        if not wait: raise FileNotFoundError('Primary blocks pending')
        time.sleep(14)
    cache=ROOT/'marginal_results'/f'panel_{panel:02d}.json'
    run=json.loads((r.OUT/'run.json').read_text())
    if cache.exists():
        prev=json.loads(cache.read_text())
        if prev['fingerprint']==run['fingerprint'] and prev['analysis']==r.v38.digest(Path(__file__)):
            return prev
    p0=json.loads(paths[0].with_suffix('.json').read_text())
    with np.load(paths[0]) as z: alpha=z['alpha']
    banks=[r.v38.Engine(p0['pair'],*aa).inner for aa in alpha]
    model=r.selector41.load_selector()
    templates={}; variances={}; rows=[]
    for (fam,hkey,var,law),path in zip(r.CONDITIONS,paths):
        key=fam+'_'+hkey; height=r.hvalue(hkey)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['pair']==p0['pair'] and meta['fingerprint']==run['fingerprint']
        if key not in templates:
            with np.load(r.direction_file(panel,fam,hkey)) as d:
                templates[key],variances[key]=items(banks,height,d['pairs'])
        with np.load(path) as a:
            target=a['labels']; z=a['z']; w=a['weights']; actions=a['actions']
            first3=a['first3_y']; fourth=a['fourth_y']; dirs=a['dirs']
            hard=np.take(a['p_base'][2],sham41.DRAWS,axis=-1)
            losses=np.empty((2,3,2,3))
            for i,sign in enumerate(r.SIGNS):
                ctx=(fam,var,height,law,sign,panel)
                data=[r.world.deployment(s,aa,ctx,r.REGIMES[0],r.REPEATS) for s,aa in zip(meta['deployment_seeds'],alpha)]
                xx,y,_,obs=zip(*data)
                np.testing.assert_array_equal(np.stack(y),target[i,0])
                sensors=[{k:np.take(v,sham41.DRAWS,axis=1) for k,v in d.items()} for d in obs]
                soft=r.decode(banks,templates[key],np.stack(xx),actions[i],sensors)
                for d,base in enumerate((hard[i],soft)):
                    for j in range(3):
                        fixed=r.v38.apply(base,(.325*z[i])[:,None,None],r.v38.BETA)
                        opt=r.v38.apply(base,(w[i,j]*z[i])[:,None,None],r.v38.BETA)
                        choose=r.selector41.select(model,base,z[i],w[i,j],actions[i],first3[i],fourth[i,j],dirs[i])
                        for k,pred in enumerate((fixed,opt,np.where(choose[:,None,:],opt,fixed))):
                            losses[i,j,d,k]=((pred-target[i,j,:,:,None])**2).mean()
            rows.append(losses.mean((0,1)).tolist())
    rec=dict(loss=rows,variance=variances,fingerprint=run['fingerprint'],analysis=r.v38.digest(Path(__file__)))
    cache.parent.mkdir(exist_ok=True)
    r.v38.atomic_json(cache,rec)
    return rec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wait',action='store_true')
    args=ap.parse_args()
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    r.torch.set_num_threads(1)
    with r.torch.inference_mode():
        data=[]
        for panel in range(r.PANELS):
            data.append(panel_rows(panel,args.wait))
            print('completed finite variance comparison',panel,flush=True)
    m=np.array([rec['loss'] for rec in data]).mean(axis=1)
    out=dict(method_order=['paired_hard','paired_finite_variance'],policies=r.POLICIES,means=m.mean(0).tolist(),
             soft_minus_hard=[sham41.interval(m[:,1,k]-m[:,0,k]) for k in range(3)],
             soft_select_minus_fixed=sham41.interval(m[:,1,2]-m[:,1,0]))
    r.v38.atomic_json(ROOT/'marginal41.json',out)

if __name__=='__main__': main()
