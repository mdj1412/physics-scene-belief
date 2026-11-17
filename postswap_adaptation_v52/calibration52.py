"""Blind action bank on old/new calibration sessions and session-drift diagnostics."""
import os
import json
import numpy as np
from scipy.stats import rankdata
from policy52 import ROOT, MODES, BUDGET, KAPPA, witnesses, load_model, decide, posterior_single

KEYS=('action','alpha','old','new','drift_prob','post_target','target_alpha')
def panel(i,model):
    shp=(5,5,len(BUDGET),4,256,2)
    arrays={k:np.empty(shp,dtype=np.int8 if k=='action' else np.float64) for k in KEYS}
    for old in range(5):
        for new in range(5):
            for n,k in enumerate(BUDGET):
                raw=witnesses(i,old,new,k)
                for m,mode in enumerate(MODES):
                    d=decide(*raw,mode,model)
                    for key in KEYS:
                        arrays[key][old,new,n,:,:,m]=d[key]
                    np.testing.assert_allclose(d['old'],posterior_single(*raw[:2],mode),atol=2e-12)
                    np.testing.assert_allclose(d['new'],posterior_single(*raw[2:],mode),atol=2e-12)
                    if k==0:
                        np.testing.assert_allclose(d['drift_prob'],.25,rtol=0,atol=2e-12)
                        assert not np.any(d['action']==1)
    path=ROOT/'decisions52'/f'panel_{i:02d}.npz'
    path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays,kappa=KAPPA,kpost=BUDGET,modes=MODES)
    tmp.replace(path)
    return arrays

def auc(negative,positive):
    """Equal frequency positive=drift, negative=preserve; ties receive one half."""
    n,m=len(negative),len(positive)
    ranks=rankdata(np.round(np.r_[negative,positive],12))
    return (ranks[n:].sum()-m*(m+1)/2)/(n*m)

def main():
    model=load_model()
    au=np.empty((12,5,5,2));cm=np.empty((12,5,5,2,2)); rate=np.empty((12,5,5,2,2,3))
    for i in range(12):
        data=panel(i,model)
        for o,n,m in np.ndindex(5,5,2):
            p=data['drift_prob'][o,o,n,...,m].ravel()
            q=data['drift_prob'][o,4-o,n,...,m].ravel()
            au[i,o,n,m]=auc(p,q)
            cm[i,o,n,m]=[(p>.5+1e-12).mean(),(q>.5+1e-12).mean()]
            for ev,post in enumerate((o,4-o)):
                actions=data['action'][o,post,n,...,m].ravel()
                rate[i,o,n,m,ev]=np.bincount(actions,minlength=3)/len(actions)
        print('cached',i,flush=True)
    out=dict(kappa=KAPPA,kpost=BUDGET,methods=MODES,
             auc_by_kappa_kpost=au.mean(0).tolist(),fpr_tpr=cm.mean(0).tolist(),
             action_freq=rate.mean(0).tolist(),action_names=['keep','refit','unsigned'],
             action_axes='pre kappa,post budget,mode,event(preserve/flip),action',
             auc_equal_kappa=au.mean((0,1)).tolist(),fpr_tpr_equal_kappa=cm.mean((0,1)).tolist(),
             scope='AUROC stratified within same catalog and pre-kappa, 1024 examples per class then average; K=0 and kappa=.5 are ties. Classification probability threshold 0.5; prior drift .25.')
    (ROOT/'calibration52.json').write_text(json.dumps(out,indent=1))
if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID')
    main()
