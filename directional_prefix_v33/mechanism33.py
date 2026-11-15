"""Paired mirror decomposition: reassign only counts across observed orientations."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent; V=ROOT.parent/'compute_matched_stopping_v32'
sys.path.insert(0,str(V))
from stats32 import allocations,predict,clean
read=lambda f:json.loads(f.read_text())
def main():
    locks=[read(V.with_name(V.name+k)/'results_train/model_lock.json') for k in ('','_rep2','_rep3')]
    laws=read(ROOT/'design_eval.json')['laws']
    out=np.zeros((4,8,3,3));opposite=np.zeros((4,8,3,2))
    for c in range(4):
        for i in range(8):
            z=[np.load(ROOT/'results_eval'/laws[2*c+d]/f'block_{i:03d}/metrics.npz') for d in range(2)]
            np.testing.assert_array_equal(z[0]['alpha'],z[1]['alpha'])
            for s,lock in enumerate(locks):
                alloc=np.asarray([[allocations(predict(lock['models'][arm],zz['features'])) for arm in ('broad','repeated')] for zz in z]);ks=np.searchsorted([2,4,8],alloc)
                D=np.array([[np.diff(zz['oracle_stage_risk'][np.arange(8)[None,:],ks[a]].mean(-1))[0] for zz in z] for a in (0,1)])
                first=((D[0,0]-D[1,0])+(D[0,1]-D[1,1]))/2
                second=((D[0,0]-D[0,1])+(D[1,0]-D[1,1]))/2
                out[c,i,s]=[first,second,D[0,0]-D[1,1]]
                opposite[c,i,s]=(ks[0]!=ks[1]).mean(-1)
            for a in z:a.close()
    assert np.max(abs(out[...,:2].sum(-1)-out[...,2]))<1e-14
    rn=np.random.default_rng(33009511);ids=rn.integers(8,size=(20000,4,8))
    draws=out[np.arange(4)[None,:,None],ids].mean((1,2,3))
    res=dict(names=['allocation','conditional_stage_risk','total'],means=out.mean((0,1,2)),
             by_seed=out.mean((0,1)),pointwise95=np.quantile(draws,[.025,.975],axis=0).T,
             fraction_environments_with_different_counts_on_reversal=opposite.mean((0,1,2)))
    idx=read(ROOT/'policy_stats33.json')['orientation']
    np.testing.assert_allclose(res['means'][2],idx[2],atol=2e-14,rtol=0)
    (ROOT/'mechanism33.json').write_text(json.dumps(clean(res),indent=2)+'\n')
if __name__=='__main__':main()
