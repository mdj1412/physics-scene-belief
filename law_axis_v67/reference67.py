"""Apply v66's archived axis and matched head initialization to the same interventions."""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.stats import t
from score67 import H,SRC,FAM,SURF,OLD,VAR,HT,LAWS, panel,name,ci,sha
def main():
    assert (H/'score.json').is_file()
    a=np.zeros((12,2,4,5))
    for k in range(12):
        pan=panel(k)
        with np.load(SRC.parent/'exp_locked_preds'/(pan+'.npz')) as z:
            cells=list(z['cells']);heads=z['pred']
        for i,f in enumerate(FAM):
            for s,v,h in itertools.product(SURF,VAR,HT):
                with np.load(SRC/'routes'/pan/f'{f}_{v}_h{h}.npz') as z:rt=z['route']
                y=[]
                for l in LAWS:
                    root=SRC if l in OLD else H
                    with np.load(root/'labels'/pan/name(f,s,l,v,h)) as z:y.append(z['labels'][1:].mean(0))
                y=np.array(y)[...,None]
                for acq in OLD:
                    fn=name(f,s,acq,v,h)
                    with np.load(SRC/'results'/pan/fn) as z:full,hard,soft=z['pred'][:,0]
                    # exp_locked.py stores [full(3), dropped(3), restored(3)]. Index 0 in each triple is seed 66101.
                    drop,restore=heads[cells.index(fn),[3,6]].astype(float)
                    methods=np.stack([full,soft,hard,drop,restore])
                    mix=np.broadcast_to(methods[:,None],(5,3,*full.shape)).copy()
                    mix[1:3]=np.where(rt[None,...,None],mix[1:3],mix[:1])
                    risk=(mix[None]**2+(1-2*mix[None])*y[:,None]).mean((2,3,4,5))
                    a[k,i]+=risk/24
    p=json.loads((H/'score.json').read_text())
    refit=np.array(p['panels']).mean(3)
    soft=a[:,:,:,1]-a[:,:,:,0]; rescue=a[:,:,:,4]-a[:,:,:,3]
    current=refit[:,:,:,1]-refit[:,:,:,0]
    c=dict(families=FAM,laws=LAWS,original_family_soft=ci(soft,6),original_family_rescue=ci(rescue,6),
      original_pooled_rescue=ci(rescue.mean(1),6),paired_refit_minus_original_soft=ci(current-soft,6),
      catalog_means=a,info='Legacy v45 axis + matched historical head seed 66101; matched acquisition tapes x terminal law')
    (H/'reference_axis.json').write_text(json.dumps(c,indent=2,default=lambda x:np.asarray(x).tolist()))
if __name__=='__main__':main()
