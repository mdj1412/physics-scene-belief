"""Secondary accounting for surface/law balance and empirical p90 in a policy lottery."""
import os
import json
from pathlib import Path
import numpy as np
from score55 import ROOT,C,G,IMP,REF

def q90(x,w):
    order=np.argsort(x)
    masses=np.cumsum(w[order]/w.sum())
    return x[order[np.searchsorted(masses,.90)]]

def main():
    assert os.environ.get('SLURM_JOB_ID')
    published=json.loads((ROOT/'score55.json').read_text())
    with np.load(ROOT/'clusters55.npz') as f:
        means=f['by_surface']; rates=f['per_catalog_rate']
    with np.load(ROOT/'bootstrap55.npz') as f:
        weights=f['catalog_counts']/C
    need={}; all_loss=[]; matched=[]; surf_map=[]; rate_per_env=[]
    for c in range(C):
        with np.load(ROOT/'worlds'/f'panel_{c:02d}.npz') as f:
            family=np.array([str(key).split('_')[1] for key in f['surface']]); surf_map.append(family)
            law=f['law']
        with np.load(ROOT/'scored55'/f'panel_{c:02d}.npz') as f:
            raw=f['real'][0]; env_rate=f['gate'].mean((0,2))
            rates2=f['rate_env']
            np.testing.assert_allclose(env_rate,rates2)
            rate_per_env.extend(env_rate)
            source=f['exact_cells'].mean((0,1))
            sub=source[:,2]+env_rate*(source[:,1]-source[:,2])
            matched.append((source[:,0]-sub).mean())
        for i in range(4):
            key=family[i]+' × '+str(law[i])
            need.setdefault(key,[]).append([*means[c,i,:3],env_rate[i]])
        all_loss.append(raw)
    group={}
    for name,rows in need.items():
        m=np.array(rows);assert len(m)==C
        diff=m[:,0,None]-m[:,[1,2]]
        group[name]=dict(gate=m[:,0].mean(),I=m[:,1].mean(),lottery=m[:,2].mean(),impact_buy_rate=m[:,3].mean(),
                         differences=diff.mean(0).tolist(),win=(diff<0).sum(0).tolist())
    boot=weights@matched
    within=dict(diff=float(np.mean(matched)),ci=np.quantile(boot,[.025,.975]).tolist())
    raw=np.array(all_loss) # C,e,event,k,env,trial,action
    ew=np.broadcast_to(np.array([1-G,G])[None,None,:,None,None,None],raw.shape[:-1]).ravel()
    r=rates.mean()
    vals=[q90(raw[...,col].ravel(),ew) for col in (0,1)]
    vals.append(q90(np.r_[raw[...,REF].ravel(),raw[...,IMP].ravel()],np.r_[ew*(1-r),ew*r]))
    np.testing.assert_allclose(vals,published['session_p90'][:3],rtol=0,atol=2e-8)
    out=dict(secondary_family_law=group,gate_minus_conditioned_lottery=within,
             fully_sorted_quantile=vals,per_env_rate_range=[min(rate_per_env),max(rate_per_env)])
    (ROOT/'audit55.json').write_text(json.dumps(out,indent=1,default=lambda x:np.asarray(x).tolist()))
    print(out)
if __name__=='__main__':main()
