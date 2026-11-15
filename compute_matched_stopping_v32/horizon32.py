"""How much can a one-stage gain target diverge from full stopping utility?"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_eval'
LAWS=('split_static','speed_weak','directional')
ALL=np.arange(8)


def menus():
    candidates=[]
    for middle in combinations(ALL,4):
        for final in combinations(middle,2):
            v=np.zeros(8,dtype=int)
            v[list(middle)]=1; v[list(final)]=2
            candidates.append(v)
    assert len(candidates)==420
    return np.array(candidates)


def choose(cost,menu):
    top=np.sort(np.argsort(cost[:,1]-cost[:,0],kind='stable')[:4])
    finalists=top[np.argsort((cost[:,2]-cost[:,1])[top],kind='stable')[:2]]
    myopic=np.zeros(8,dtype=int);myopic[top]=1;myopic[finalists]=2
    full=menu[np.argmin(cost[ALL,menu].mean(1))]
    assert cost[ALL,full].mean()<=cost[ALL,myopic].mean()+1e-14
    return np.stack([myopic,full])


def main():
    menu=menus()
    bylaw=[]
    for law in LAWS:
        rows=[];expected=None
        for p in sorted((OUT/law).glob('block_*/metrics.npz')):
            with np.load(p) as z:
                a=z['alpha'];ix=np.lexsort((a[:,1],a[:,0]))
                if expected is None: expected=a[ix]
                np.testing.assert_array_equal(expected,a[ix])
                rows.append(z['oracle_stage_risk'][ix])
        assert len(rows)==16
        row=np.stack(rows);out=[]
        for i,r in enumerate(row):
            other=np.delete(row,i,axis=0).mean(0)
            decisions=np.concatenate([choose(r,menu),choose(other,menu)],axis=0)
            assert (np.array([2,4,8])[decisions].sum(1)==32).all()
            out.append([*r[ALL,decisions].mean(1),r[:,1].mean()])
        bylaw.append(np.mean(out,axis=0))
    obj=dict(order=['same_block_myopic','same_block_full','separate_blocks_myopic','separate_blocks_full','fixed4'],
             nominal_budgets=[2,4,8],by_law=bylaw,pooled=np.mean(bylaw,axis=0),laws=LAWS)
    (OUT/'horizon32.json').write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist()))
if __name__=='__main__':main()
