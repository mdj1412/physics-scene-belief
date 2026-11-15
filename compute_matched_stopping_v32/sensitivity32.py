"""Frozen sensitivity: disjoint source fits with unchanged features, penalty and target blocks."""
import json


import numpy as np

from fit32 import PENALTY, read_arm
from stats32 import (ROOT, METHODS, SEED, clean, allocations, load_results, predict, sample_ids)


def ridge_svd(x, y):
    m = x.mean(0); sd=np.maximum(x.std(0),1e-8)
    xx=(x-m)/sd
    mat=np.vstack((xx, np.sqrt(len(y)*PENALTY)*np.eye(x.shape[1])))
    rhs=np.r_[y-y.mean(),np.zeros(x.shape[1])]
    return dict(mean=m,sd=sd,beta=np.linalg.lstsq(mat,rhs,rcond=None)[0],intercept=y.mean(),lam=PENALTY)


def fits(root):
    models={}
    for arm in METHODS[:2]:
        x,y,metadata=read_arm(root/'results_train',arm,15)
        block=np.repeat(np.arange(len(metadata)),[m['n_environments'] for m in metadata])
        models[arm]={}
        for n in (2,3):
            for k in range(n):
                use=block % n==k
                label_s=sum(m['elapsed_s'] for m in metadata if m['block'] % n==k)
                models[arm][n,k]=( [ridge_svd(s[use],t[use]) for s,t in zip(x,y)],
                                    {'n_environments': int(use.sum()), 'label_seconds':label_s,
                                     'block_ids':np.unique(block[use]).tolist()})
    return models


def analyze_subsets(root=ROOT):
    models=fits(root)
    _, blocks=load_results(root)
    draws=sample_ids(16,np.random.default_rng(SEED))
    ids=np.arange(3)[None,:,None]
    records=[]
    for n,k in models['broad']:
        risks=[];changed=[]
        for law in blocks:
            law_risks=[];law_changed=[]
            for block in law:
                r=[];c=[]
                for i,a in enumerate(METHODS[:2]):
                    counts=allocations(predict(models[a][n,k][0],block['x']))
                    r.append(block['risk'][np.arange(8),np.searchsorted([2,4,8],counts)].mean())
                    c.append(np.mean(counts!=block['counts'][:,i]))
                law_risks.append([*r,*block['brier'][2:]]);law_changed.append(c)
            risks.append(law_risks);changed.append(law_changed)
        risks=np.asarray(risks)
        result=risks.mean(axis=(0,1)); paired=(risks[...,1]-risks[...,0])
        stars=paired[ids,draws].mean(axis=(1,2))
        records.append({'modulus':n,'residue':k,'brier':result,'repeated_minus_broad':result[1]-result[0],
              'paired_conditional_ci':np.quantile(stars,[.025,.975]),'by_law_diff':paired.mean(axis=1),
              'allocation_change_from_full':np.mean(changed,axis=(0,1)),
              'source':{a:models[a][n,k][1] for a in METHODS[:2]}})
    return clean({'methods':METHODS,'subsets':records, 'note':'interleaved disjoint blocks within modulus, overlapping moduli; reduced training time conditional sensitivity'})


if __name__=='__main__':
    out=ROOT/'results_eval'/'source_subset_sensitivity32.json'
    result=analyze_subsets();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print([(r['modulus'],r['residue'],r['repeated_minus_broad']) for r in result['subsets']])
