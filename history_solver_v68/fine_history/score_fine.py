"""Paired 2×2 quadrature comparison plus unidentified-label envelopes."""
import itertools
import json
import os
from pathlib import Path
import sys
import numpy as np
H=Path(__file__).resolve().parent
M=H.parent
sys.path.insert(0,str(M))
from crossover68 import archive,routed,ROUTES
from unresolved_bounds68 import envelope
from score_main68 import summarize,digest
F=('wendland','dispersive')
L=('strengthening','weakening')
def pack(x):
    # x: catalogs, family, law, seed, stage
    d=x[...,1]-x[...,0]
    a=x[...,4]-x[...,3]
    return np.stack((d[:,0],d[:,1],a.mean(1)),axis=-1)
def contrasts(w):
    return {l:summarize(w[:,b].mean(1),6).tolist() for b,l in enumerate(L)}
def run():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    se=json.loads((H/'FORECASTS_SEALED').read_text()); assert digest(M/'FORECASTS_SEALED')==se['main_seal']
    for x,sha in se['outputs'].items():assert digest(H/x)==sha
    aps=np.zeros((12,2,2,2,2,4,5)) # catalog, hist resolution, terminal resolution, family, law,seed,stage
    envs=np.zeros((12,2,2,2,2))
    coverage=np.zeros((12,2,2,2))
    switches=np.zeros((12,2,2,3))
    for k in range(12):
        pan=f'panel_{k:02}';c=archive(M/'forecasts'/'compliant'/(pan+'.npz'));f=archive(H/'forecasts'/'compliant'/(pan+'.npz'))
        with np.load(M/'forecasts'/'compliant'/(pan+'.npz')) as z,np.load(H/'forecasts'/'compliant'/(pan+'.npz')) as t:
            np.testing.assert_array_equal(z['cells'],t['cells'])
            act_c=z['actions'];act_f=t['actions']
        for name,idx in c['ix'].items():
            family,s,law,v,h=name[:-4].split('_');fi=F.index(family);li=L.index(law)
            with np.load(ROUTES/pan/f'{family}_{v}_{h}.npz') as z:route=z['route']
            with np.load(M/'labels_compliant'/pan/name) as z:
                yc=z['labels'].astype(float)
            with np.load(M/'labels_compliant_refined'/pan/name) as z:
                yf=z['labels'].astype(float);unres=z['unresolved']
            coverage[k,fi,li]+=[np.mean(unres)/12,np.mean(yc!=yf)/12]
            pf=None
            for e,rec in enumerate((c,f)):
                p=routed(rec['p'][idx],rec['shift'][idx],route)
                for a,y in enumerate((yc,yf)):
                    loss=(p*p+y.mean(0)[None,None,...,None]*(1-2*p)).mean((2,3,4,5))
                    aps[k,e,a,fi,li]+=loss/12
                if e:pf=p
            switches[k,fi,li]+=[np.mean(act_c[idx]!=act_f[idx])/12,abs(c['shift'][idx]-f['shift'][idx]).mean()/12,abs(c['p'][idx]-f['p'][idx]).mean()/12]
            for j,(a,b) in enumerate(((1,0),(4,3))):
                # Binary label toggles shift method contrast exactly by -2 times their mean probability difference.
                slope=-2*(pf[:,a]-pf[:,b]).mean((0,4))[None]
                actual=(loss[:,a]-loss[:,b]).mean()
                lower=actual+np.mean(unres*(np.minimum(0.,slope)-slope*yf))
                upper=actual+np.mean(unres*(np.maximum(0.,slope)-slope*yf))
                envs[k,fi,li,j]+=[lower/12,upper/12]
    opts={'coarse/coarse':pack(aps[:,0,0]),'coarse/fine':pack(aps[:,0,1]),
          'fine/coarse':pack(aps[:,1,0]),'fine/fine':pack(aps[:,1,1])}
    cs={k:contrasts(v) for k,v in opts.items()}
    comparisons={}
    for a,b in [('fine/fine','coarse/coarse'),('fine/fine','coarse/fine'),('coarse/fine','coarse/coarse')]:
        comparisons[a+' minus '+b]=contrasts(opts[a]-opts[b])
    bounds={}
    for i,l in enumerate(L):
        for j,a in enumerate([envs[:,0,i,0],envs[:,1,i,0],envs[:,:,i,1].mean(1)]):
            bounds[l+'_'+str(j)]=envelope(a[:,0],a[:,1])
    primary=json.loads((M/'score_main68.json').read_text())['catalog_absolute']
    err=float(np.max(np.abs(aps[:,0,0]-np.array(primary)[:,1,:,:,2])))
    assert err<1e-13
    data=dict(contrasts=cs,paired_changes=comparisons,full_fine_unknown_envelopes=bounds,abs_axes='catalog x history x terminal x family x law x axis x stage',
        absolute=aps.mean((0,5)).tolist(),catalogs=aps.tolist(),
        other_by_family_law=dict(unknown_rate_and_label_disagreement=coverage.mean(0).tolist(),acquisition_switch_mean_shift_change_and_mean_abs_probability_change=switches.mean(0).tolist()),
        original_replay_error=err,sites=[9,17],seeds=list(range(670101,670105)),job=os.environ['SLURM_JOB_ID'])
    (H/'fine_scores.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':run()
