"""Geometry baseline using only the three nearest coordinates, with zero passive pairs."""
import json
import os
import time
from pathlib import Path
import numpy as np
import runner41 as r
from sham41 import DRAWS,interval

def run_panel(panel):
    b=panel%2
    paths=[r.rawpath(c,panel,b) for c in range(24)]
    while not all(p.with_suffix('.json').exists() for p in paths): time.sleep(15)
    first=json.loads(paths[0].with_suffix('.json').read_text())
    cache=r.ROOT/'local_window_results'/f'panel_{panel:02d}.json'
    if cache.exists():
        rec=json.loads(cache.read_text())
        if rec['fingerprint']==first['fingerprint'] and rec['analysis']==r.v38.digest(Path(__file__)):
            return rec['rows']
    with np.load(paths[0]) as f: alpha=f['alpha']
    banks=[r.v38.Engine(first['pair'],*aa).inner for aa in alpha]
    templates={}
    for hkey in r.HEIGHTS:
        items=[]
        for mean,_,label in r.templates(banks,r.hvalue(hkey)):
            local=mean.clone()
            local[:,[0,1,5,6]]=0
            items.append((local,.5*local.square().sum(-1),label))
        templates[hkey]=items
    rows=[]
    for (family,hkey,variant,law),path in zip(r.CONDITIONS,paths):
        height=r.hvalue(hkey)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert first['pair']==meta['pair'] and first['fingerprint']==meta['fingerprint']
        with np.load(path) as f:
            target=f['labels'];z=f['z'];action=f['actions']
            raw=np.take(f['p_base'],DRAWS,axis=-1)
        losses=[]
        for i,sign in enumerate(r.SIGNS):
            ctx=(family,variant,height,law,sign,panel)
            data=[r.world.deployment(seed,aa,ctx,r.REGIMES[0],r.REPEATS) for seed,aa in zip(meta['deployment_seeds'],alpha)]
            xx,truth,_,obs=zip(*data)
            np.testing.assert_array_equal(np.stack(truth),target[i,0])
            sensors=[{k:np.take(v,DRAWS,axis=1) for k,v in o.items()} for o in obs]
            local=r.decode(banks,templates[hkey],np.stack(xx),action[i],sensors)
            base=np.concatenate([raw[:,i],local[None]],axis=0)
            fixed=r.v38.apply(base,.325*z[i][None,:,None,None],r.v38.BETA)
            losses.append(np.square(fixed[:,None]-target[i][None,:,:,:,None]).mean(axis=(1,2,3,4)))
        rows.append(np.mean(losses,axis=0).tolist())
    cache.parent.mkdir(exist_ok=True)
    r.v38.atomic_json(cache,dict(analysis=r.v38.digest(Path(__file__)),fingerprint=first['fingerprint'],rows=rows))
    return rows

def main():
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    r.torch.set_num_threads(1)
    with r.torch.inference_mode(): rows=np.array([run_panel(p) for p in range(12)])
    v=rows.mean(1)
    groups={}
    for family in r.world.FAMILIES:
        for hkey in r.HEIGHTS:
            idx=[i for i,item in enumerate(r.CONDITIONS) if item[:2]==(family,hkey)]
            sub=rows[:,idx].mean(1)
            groups[family+'_'+hkey]=interval(sub[:,2]-sub[:,3])
    r.v38.atomic_json(r.ROOT/'local_window41.json',dict(order=[*r.DECODERS,'local_window'],mean=v.mean(0).tolist(),
                      estimated_minus_local=interval(v[:,2]-v[:,3]),by_shape_height=groups))
if __name__=='__main__': main()
