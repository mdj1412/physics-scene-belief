"""Matched negative control, one preselected block and two noise draws per panel."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.stats import t
import runner41 as r

ROOT=Path(__file__).resolve().parent
DRAWS=[0,16]
def interval(x):
    a=np.asarray(x)
    error=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return [float(a.mean()), float(a.mean()-error),float(a.mean()+error)]

def panel_rows(panel, wait):
    b=panel%2
    paths=[r.rawpath(i,panel,b) for i in range(len(r.CELLS))]
    while not all(p.with_suffix('.json').exists() for p in paths):
        if not wait: raise FileNotFoundError('Primary blocks pending')
        time.sleep(12)
    cache=ROOT/'sham_results'/f'panel_{panel:02d}.json'
    run=json.loads((r.OUT/'run.json').read_text())
    if cache.exists():
        record=json.loads(cache.read_text())
        if record['fingerprint']==run['fingerprint'] and record['analysis']==r.v38.digest(Path(__file__)):
            return record['loss']
    first=json.loads(paths[0].with_suffix('.json').read_text())
    with np.load(paths[0]) as a: alpha=a['alpha']
    banks=[r.v38.Engine(first['pair'],*aa).inner for aa in alpha]
    templates={}
    rows=[]
    for condition,path in zip(r.CONDITIONS,paths):
        fam,hkey,variant,law=condition
        height=r.hvalue(hkey); key=(fam,hkey)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['fingerprint']==run['fingerprint'] and meta['pair']==first['pair']
        if key not in templates:
            with np.load(r.direction_file(panel,*key)) as d: pairs=d['pairs'].copy()
            pairs[:,1]=np.roll(pairs[:,1],1,axis=0)
            templates[key]=r.templates(banks,height,r.world.estimate_direction(pairs))
        with np.load(path) as a:
            np.testing.assert_array_equal(alpha,a['alpha'])
            target=a['labels']
            choices=a['actions']; z=a['z']
            base=np.take(a['p_base'],DRAWS,axis=-1)
            shuffled=[]
            for i,sign in enumerate(r.SIGNS):
                context=(fam,variant,height,law,sign,panel)
                rows_deploy=[r.world.deployment(s,aa,context,r.REGIMES[0],repeats=r.REPEATS)
                             for s,aa in zip(meta['deployment_seeds'],alpha)]
                xx,y,_,obs=zip(*rows_deploy)
                np.testing.assert_array_equal(np.stack(y),target[i,0])
                sensors=[{k:np.take(v,DRAWS,axis=1) for k,v in item.items()} for item in obs]
                shuffled.append(r.decode(banks,templates[key],np.stack(xx),choices[i],sensors))
            pred=np.concatenate([base,np.asarray(shuffled)[None]],axis=0)
            prob=r.v38.apply(pred,.325*z[None,:,:,None,None],r.v38.BETA)
            assert prob.shape==(4,2,8,128,2)
            rows.append(np.square(prob[:,:,None]-target[None,:,:,:,:,None]).mean(axis=(1,2,3,4,5)).tolist())
    cache.parent.mkdir(exist_ok=True)
    r.v38.atomic_json(cache,dict(analysis=r.v38.digest(Path(__file__)),fingerprint=run['fingerprint'],loss=rows))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wait',action='store_true')
    args=ap.parse_args()
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    r.torch.set_num_threads(1)
    data=[]
    with r.torch.inference_mode():
        for p in range(r.PANELS):
            data.append(panel_rows(p,args.wait))
            print('completed CPU negative control panel',p,flush=True)
    cell=np.array(data)
    means=cell.mean(1)
    groups={}
    for f in r.world.FAMILIES:
        for h in r.HEIGHTS:
            ids=[i for i,row in enumerate(r.CONDITIONS) if row[:2]==(f,h)]
            v=cell[:,ids].mean(1)
            groups[f+'_'+h]=dict(brier=v.mean(0).tolist(),paired_minus_sham=interval(v[:,2]-v[:,3]))
    r.v38.atomic_json(ROOT/'sham41.json',dict(decoders=[*r.DECODERS,'unpaired'],noise_indices=DRAWS,
        n_panels=r.PANELS,mean=means.mean(0).tolist(),paired_minus_sham=interval(means[:,2]-means[:,3]),
        sham_minus_original=interval(means[:,3]-means[:,0]),groups=groups))
if __name__=='__main__': main()
