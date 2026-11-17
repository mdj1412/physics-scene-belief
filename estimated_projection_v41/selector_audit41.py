"""Verify selection at runtime and compare to equal-frequency blind mixing within each cell."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
from selector41 import ROOT,design,select,load_selector

M=('take','gap','route','marginal','reuse_original','max_probability','oracle','vs_if_opt','vs_if_fix',
   'win_share','misrank','switch_vs_original')


def interval(x):
    n=len(x);v=x.mean(0);r=t.ppf(.975,n-1)*x.std(0,ddof=1)/np.sqrt(n)
    return dict(mean=v.tolist(),ci=np.stack((v-r,v+r),axis=-1).tolist())


def read_block(path,model,cell):
    with np.load(path) as f:
        a={k:f[k] for k in ('risk','chosen_optimizer','p_base','actions','z','weights','first3_y','fourth_y','dirs','labels')}
    primary=a['risk']
    out=[]
    for j in range(3):
        part=[]
        for s in range(2):
            for r in range(3):
                p=a['p_base'][j,s];y=a['labels'][s,r,:,:,None]
                args=(a['z'][s],a['weights'][s,r],a['actions'][s],a['first3_y'][s],a['fourth_y'][s,r],a['dirs'][s])
                _,weight,h,mid=design(p,*args)
                gain=-2*weight*(h*(y-mid)).sum(1)/np.maximum(h.sum(1),1e-15)
                chosen=a['chosen_optimizer'][s,r,j]
                np.testing.assert_array_equal(select(model,p,*args),chosen)
                np.testing.assert_allclose(gain.mean(1),primary[s,r,j,1]-primary[s,r,j,0],atol=2e-12)
                np.testing.assert_allclose((chosen*gain).mean(1),primary[s,r,j,2]-primary[s,r,j,0],atol=2e-12)
                higher=(a['z'][s]*(a['weights'][s,r]-.325)>0)[:,None]
                other=a['chosen_optimizer'][s,r,0]
                part.append([chosen.mean(),gain.mean(),(chosen*gain).mean(),np.nan,
                    (other*gain).mean(),(higher*gain).mean(),gain.clip(max=0).mean(),
                    (chosen*gain).mean(),((1-chosen)*gain).mean(),(gain<0).mean(),(chosen!=(gain<0)).mean(),
                    (chosen!=other).mean()])
        out.append(np.mean(part,axis=0))
    return out


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text())
    assert meta['blocks']==576
    model=load_selector();cells=meta['cells']
    vals=np.empty((12,len(cells),2,3,len(M)))
    for c,cell in enumerate(cells):
        for j in range(12):
            for b in range(2):
                path=ROOT/'results'/cell/f'catalog_{j:02d}'/f'block_{b:02d}.npz'
                vals[j,c,b]=read_block(path,model,cell)
    # All orientations/regimes, environments, noise and both starts within each prespecified cell-panel.
    panels=vals.mean(2)
    panels[:,:,:,3]=panels[:,:,:,0]*panels[:,:,:,1]
    metrics={}
    for j,name in enumerate(meta['decoders']):
        v=panels[:,:,j].mean(1)
        desc=dict((k,float(v[:,i].mean())) for i,k in enumerate(M))
        desc['mean_gap_if_selected']=float(v[:,7].mean()/v[:,0].mean()) if v[:,0].mean() else None
        desc['mean_gap_if_fixed']=float(v[:,8].mean()/(1-v[:,0].mean())) if v[:,0].mean()<1 else None
        desc['vs_blind']=interval(v[:,2]-v[:,3])
        desc['vs_original_choice']=interval(v[:,2]-v[:,4])
        desc['vs_v39_choice']=interval(v[:,2]-v[:,5])
        desc['regret_vs_oracle']=interval(v[:,2]-v[:,6])
        desc['optimizer_fraction_ci']=interval(v[:,0])
        metrics[name]=desc
    raw=dict(metrics=metrics,cell_names=cells,columns=M,cell_average=panels.mean(0).tolist())
    (ROOT/'selector_audit41.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
