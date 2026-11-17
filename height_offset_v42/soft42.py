"""Supplementary finite-variance geometry likelihood frozen from v41, same subset as reveal."""
import json
import os
import time
import numpy as np
import runner42 as job
import stats42 as s
r=job.r
import marginal41

def panel(n):
    paths=[s.raw(row,n) for row in s.CELLS]
    while not all(p.with_suffix('.json').exists() for p in paths):time.sleep(6)
    meta=json.loads(paths[0].with_suffix('.json').read_text())
    with np.load(paths[0]) as a: alpha=a['alpha']
    banks=[r.v38.Engine(meta['model_bank_seed'],*aa).inner for aa in alpha]
    templates={};rows=[]
    for row,path in zip(s.CELLS,paths):
        fam,hk,var,law=row; h=r.hvalue(hk)
        if (fam,hk) not in templates:
            with np.load(job.OUT/'directions'/f'panel_{n:02d}_{fam}_{hk}.npz') as d:
                templates[fam,hk],_=marginal41.items(banks,h,d['pairs'])
        with np.load(path) as a:
            action=a['actions'][:,:,:32]; base=a['p_base'][:,:,:,:32,0]; targets=a['labels'][:,:,:,:,:32]
        prob=[]
        for si,sign in enumerate(r.SIGNS):
            ctx=(fam,var,h,law,sign,job.w.catalog_panel(n))
            data=[r.world.deployment(seed,aa,ctx,r.REGIMES[0],job.REPEATS)
                  for seed,aa in zip(meta['deployment_seeds'],alpha)]
            xx,truth,_,obs=zip(*data)
            np.testing.assert_array_equal(np.stack(truth)[:,:32],targets[s.Z,si,0])
            sensors=[{a:ys[:32,:1] for a,ys in d.items()} for d in obs]
            prob.append(r.decode(banks,templates[fam,hk],np.stack(xx)[:,:32],action[si],sensors)[:,:,0])
        p=np.concatenate((base,np.array(prob)[None]))
        rows.append(np.square(p[None,:,:,None]-targets[:,None]).mean((2,3,4,5)))
    print('finite variance panel',n,flush=True)
    return rows

def main():
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    r.torch.set_num_threads(1)
    with r.torch.inference_mode():loss=np.array([panel(i) for i in range(12)])
    p=loss.mean(1)
    out=dict(order=[*r.DECODERS,'finite_variance'],risk=p.mean(0),soft_minus_hard=s.ci(p[:,:,3]-p[:,:,2],9),
             soft_minus_original=s.ci(p[:,:,3]-p[:,:,0],9),nonzero=s.ci(p[:,s.NZ,:].mean(1)))
    (job.ROOT/'soft42.json').write_text(json.dumps(out,indent=2,default=lambda a:a.tolist(),allow_nan=False)+'\n')
if __name__=='__main__':main()
