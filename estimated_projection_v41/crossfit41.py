"""Prospective within-environment label-transfer diagnostic on distinct prototype folds."""
import json
from pathlib import Path
import numpy as np
from selector41 import ROOT,BETA
from selector_audit41 import interval


def splits(seed):
    # Same ID replay as world41, then a distinct RNG for splitting unique prototype IDs.
    ids=np.random.default_rng(np.random.SeedSequence([seed,270000,4120])).integers(2048,size=128)
    unique,inv=np.unique(ids,return_inverse=True)
    rng=np.random.default_rng(np.random.SeedSequence([seed,410055]))
    result=[]
    for _ in range(8):
        pos=rng.permutation(len(unique))
        first=np.zeros(len(unique),bool)
        first[pos[:len(pos)//2]]=True
        mask=first[inv]
        assert not np.intersect1d(ids[mask],ids[~mask]).size
        result.append(mask)
    return np.array(result)


def block(path):
    meta=json.loads(path.with_suffix('.json').read_text())
    masks=np.stack([splits(seed) for seed in meta['deployment_seeds']])
    n=masks.sum(-1)[:,:,None]
    with np.load(path) as arc:
        pb=arc['p_base'];z=arc['z'];w=arc['weights'];labels=arc['labels'];picked=arc['chosen_optimizer']
    out=np.empty((3,2,3,6))
    for j in range(3):
        for s in range(2):
            for r in range(3):
                p=pb[j,s]
                odds=np.exp(BETA*z[s]*.325)[:,None,None]
                pf=p*odds/(1-p+p*odds)
                odds=np.exp(BETA*z[s]*w[s,r])[:,None,None]
                po=p*odds/(1-p+p*odds)
                diff=(po-pf)*(po+pf-2*labels[s,r,:,:,None])
                loss=diff.mean(1)
                first=np.einsum('esp,epr->esr',masks,diff)
                second=diff.sum(1)[:,None,:]-first
                cross=((second<0)*first+(first<0)*second)/128
                policy=(picked[s,r,j]*loss).mean()
                oracle=np.minimum(0,loss).mean()
                out[j,s,r]=[cross.mean(),loss.mean(),policy,oracle,
                            ((first<0)==(second<0)).mean(),(((second<0)*n+(first<0)*(128-n))/128).mean()]
    return out.mean((1,2))


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text())
    assert meta['blocks']==576
    arr=[]
    for p in range(12):
        rows=[]
        for c in meta['cells']:
            for b in range(2):
                rows.append(block(ROOT/'results'/c/f'catalog_{p:02d}'/f'block_{b:02d}.npz'))
        arr.append(np.mean(rows,axis=0))
    arr=np.asarray(arr);out={}
    for j,k in enumerate(meta['decoders']):
        cross,always,policy,oracle,agree,use=arr[:,j,:].T
        out[k]={n:interval(q) for n,q in [('crossfit_minus_fixed',cross),('crossfit_minus_optimizer',cross-always),
                ('selector_minus_crossfit',policy-cross),('same_batch_oracle_minus_crossfit',oracle-cross),
                ('half_sign_agreement',agree),('opt_fraction',use)]}
    (ROOT/'crossfit41.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
