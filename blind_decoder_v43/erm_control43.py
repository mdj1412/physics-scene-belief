"""Supplementary nominal-only ERM, fixed by source_fit43 protocol before OOD evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

import model43 as model
from source_fit43 import fit_residual, brier, save

ROOT = Path(__file__).resolve().parent


def bounds(v, comparisons=1):
    n=len(v)
    half=t.ppf(1-.025/comparisons,n-1)*v.std(ddof=1)/np.sqrt(n)
    return [float(v.mean()),float(v.mean()-half),float(v.mean()+half)]


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/'source_matrix43.npz') as src:
        x=src['feature']
        y=src['label'].mean(axis=1)
        panel=src['panel']
    fit=panel<8
    choose=panel>=10
    robust=model.load_model()
    prior=json.loads((ROOT/'source_validation43.json').read_text())
    erm=fit_residual(x[fit], y[fit,0], robust['width'])
    save('nominal_erm43.json',erm)
    sha=hashlib.sha256((ROOT/'nominal_erm43.json').read_bytes()).hexdigest()
    pr=model.robust_from_features(robust,x[choose])
    pe=model.robust_from_features(erm,x[choose])
    check = [brier(pr,y[choose,j])-brier(pe,y[choose,j]) for j in (0,1)]
    expected=[prior['source_nominal_robust_minus_erm'],prior['source_shifted_robust_minus_erm']]
    np.testing.assert_allclose(check,expected,atol=1e-9,rtol=0)
    rows=[]
    largest=0.
    for panel in range(12):
        files=sorted((ROOT/'results'/f'panel_{panel:02d}').glob('*.npz'))
        if len(files)!=48:raise RuntimeError(f'Incomplete target results: panel {panel}, {len(files)}')
        vals=[]
        for f in files:
            with np.load(f) as raw:
                pred=raw['pred']
                actions=raw['selected']
                xx=raw['xx']
                q,p=raw['query_paid'],raw['probe_paid']
                paid=[{1:qq,3:pp} for qq,pp in zip(q,p)]
                x=model.features(pred[0,0],xx,actions,paid,float(raw['height']))
                ermp=model.robust_from_features(erm,x)
                regenerated=model.robust_from_features(robust,x)
                largest=max(largest,float(abs(regenerated-pred[1,0]).max()))
                offsets=raw['offsets_mm']
                labels=raw['labels']
            scores=np.array([np.square(z[None]-labels[...,None]).mean((1,2,3)) for z in (pred[0,0],ermp,pred[1,0])])
            vals.append(np.stack([scores[:,offsets==0].mean(1),scores[:,offsets!=0].mean(1)]))
        rows.append(np.mean(vals,axis=0))
    assert largest<2e-9,largest
    a=np.asarray(rows)
    d=a[:,:,2]-a[:,:,1]
    ers=a[:,:,1]-a[:,:,0]
    out=dict(estimand='zero correction, robust minus source nominal-only ERM',
             decoder_order=['original','nominal_only_erm','augmented_robust'],
             strata=['nominal','shifted'], panel_risks=a.tolist(), mean_risks=a.mean(0).tolist(),
             robust_minus_erm=[bounds(d[:,j]) for j in range(2)],
             simultaneous_robust_minus_erm=[bounds(d[:,j],2) for j in range(2)],
             erm_minus_original=[bounds(ers[:,j]) for j in range(2)],
             source_validation_recomputed=check, source_validation_expected=expected,
             largest_robust_replay_error=largest,nominal_erm_sha256=sha, panelloss_wins=(d<0).sum(0).tolist())
    save('erm_control43.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='panel_risks'}))


if __name__=='__main__':main()
