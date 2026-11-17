"""Freeze the two training tables, then check held-out folds, decoder resolution and readout."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t
from sign_bins54 import ROOT,V53,V52,N,G,read_panel,quantiles
from score53 import quadratic, secondary_deployment, weighted_cdf_inverse
from optical53 import readings
from policy53 import witnesses53, mixed53

def main():
    assert os.environ.get("SLURM_JOB_ID")
    cfg=json.loads((ROOT/'sign_bins54.json').read_text())
    splits=cfg['splits']
    outputs={name:[] for name in ['original','replay','ultra_mean','camera']}
    for i in range(N):
        filename=f'panel_{i:02d}.npz'
        _,_,group=read_panel(i)
        rule = next(np.array(s['impact_rule']) for s in splits if i in s['test_catalogs'])
        route=rule[group]
        with np.load(V52/'risk52'/filename) as f:
            exact,real=f['exact'],f['realized']
        sec=secondary_deployment(i)
        optical=readings(i)
        camera_alpha=np.zeros((3,2,5,4,256))
        env,rep=np.indices((4,256))
        for e,ev,k in np.ndindex(3,2,5):
            ow,of,nw,nf,sign,_=witnesses53(i,k,4-k if ev else k,e)
            observed=optical[sign+1,1,env,rep]
            camera_alpha[e,ev,k]=mixed53(ow,of,nw[...,:3],nf[...,:3],nw[...,3],observed)['alpha_joint']
        with np.load(V53/'decisions53'/filename) as f:
            names=list(f['methods'].astype(str))
            a4,ai=np.moveaxis(f['alpha'][..., [names.index('reference4_joint'),names.index('impact_joint')]],-1,0)
        for name,base,obs in [('original',exact[0,...,None,:],ai),('replay',sec,ai),
                             ('ultra_mean',exact[1,...,None,:],ai),('camera',exact[0,...,None,:],camera_alpha)]:
            arms=np.stack([np.where(route,obs,a4),a4,obs],-1)
            loss=quadratic(base[...,None,:],arms,False)
            mu=((1-G.mean())*loss[:,0]+G.mean()*loss[:,1]).mean((0,1,2,3))
            raw=quadratic((sec if name=='replay' else real)[...,None,:],arms,False)
            outputs[name].append((mu,raw.astype('float32')))
    summary={}
    for name,panels in outputs.items():
        mus,raws=[np.stack(a) for a in zip(*panels)]
        block={}
        for tag,part in [('all',slice(None)),('trainA_testB',slice(6,12)),('trainB_testA',slice(0,6))]:
            mu=mus[part]
            diff=mu[:,0,None]-mu[:,1:]
            d=diff.mean(0); half=t.ppf(.975,len(mu)-1)*diff.std(0,ddof=1)/np.sqrt(len(mu))
            entry={'mean':mu.mean(0).tolist(),'mean_diff':d.tolist(),
                   'per_catalog_wins':(diff<0).sum(0).tolist()}
            if tag!='all':entry['conditional_95_mean_ci']=np.stack([d-half,d+half],axis=-1).tolist()
            if name!='ultra_mean':entry['session_p90']=quantiles(raws[part],G.mean()).tolist()
            block[tag]=entry
        summary[name]=block
    (ROOT/'validate_bins54.json').write_text(json.dumps(summary,indent=2))
    for k,v in summary.items():print(k,v['all'],flush=True)
if __name__=='__main__': main()
