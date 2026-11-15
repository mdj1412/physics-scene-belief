"""Exploratory, after primary results: hindsight allocation under existing constraints vs random/transfer."""
from pathlib import Path
import json,sys,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'compute_matched_stopping_v32'))
from stats32 import clean
def main():
    laws=json.loads((ROOT/'design_eval.json').read_text())['laws']
    menus=np.asarray([[2 if j in finals else 1 if j in first else 0 for j in range(8)]
        for first in itertools.combinations(range(8),4) for finals in itertools.combinations(first,2)])
    assert menus.shape==(420,8)
    raw=np.empty((4,8,2,5))
    for c in range(4):
        for d in range(2):
            risks=[]; bases=[]
            for i in range(8):
                with np.load(ROOT/'results_eval'/laws[2*c+d]/f'block_{i:03d}'/'metrics.npz') as z:
                    alpha=z['alpha'];idx=np.lexsort((alpha[:,1],alpha[:,0]))
                    risks.append(z['oracle_stage_risk'][idx]);bases.append(z['brier'].mean(0))
            arr=np.array(risks)
            m=arr[:,np.arange(8)[None,:],menus].mean(axis=-1)
            for i in range(8):
                j=np.delete(m,i,axis=0).mean(0).argmin()
                raw[c,i,d]=[bases[i][2],bases[i][3],m[i].mean(),m[i].min(),m[i,j]]
    draws=np.random.default_rng(3306119).integers(8,size=(10000,4,8))
    s=raw[np.arange(4)[None,:,None],draws].mean((1,2,3));v=raw.mean((0,1,2))
    report=dict(methods=['fixed4','uncertainty','uniform menu','hindsight menu','leave-one-block-out menu'],
            brier=v,vs_fixed=v-v[0],vs_fixed_pointwise95=np.quantile(s-s[:,0,None],[.025,.975],axis=0).T,
            note='Last interval does not include rerunning selection on alternative training blocks. Overlapping folds; point estimate only in report.')
    (ROOT/'headroom33.json').write_text(json.dumps(clean(report),indent=2)+'\n')
if __name__=='__main__':main()
