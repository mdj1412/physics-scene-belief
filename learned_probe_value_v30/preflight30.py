"""CPU-only physical invariants and data-integrity checks, with no fitting or inference."""
import json

import numpy as np

from contact30 import (LAWS, QUERY_Y, QUERY_INCOMING, calibration_data, collision, draw_theta,
                       evaluation_data, factual, impact, means, torch, V29)
from score30 import ROOT, features, load_training, quota_actions
from runner29 import atomic_json

assert not torch.cuda.is_available()
rows, truth, source_hashes = load_training()
assert rows.shape==(24,8,2,15) and truth.shape==(24,8,2)
assert np.isfinite(rows).all() and np.isfinite(truth).all()
d=json.loads((ROOT/'design.json').read_text())
# These lane seeds are excluded from all training and test data.
measurements={}
for law in LAWS:
    totals=[]
    for i,alpha in enumerate(d['alphas']):
        n=256;seed=830000+i
        theta=draw_theta(n,seed,20,alpha)
        source,label0=impact(theta,QUERY_Y,QUERY_INCOMING)
        fixed,label1=collision(theta,QUERY_Y,QUERY_INCOMING,"coulomb")
        assert torch.equal(label0,label1) and torch.equal(source,fixed)
        labels,source_labels,obs,(xf,yf)=evaluation_data(n,seed,alpha,law,repeats=2)
        assert np.array_equal(label0.cpu().numpy(),source_labels)
        assert np.equal(xf,0.).all() and np.all(yf>0)
        assert np.array_equal(factual(n,seed,20,alpha,law),factual(n,seed,20,alpha,"coulomb"))
        totals.append([np.mean(labels!=source_labels),np.mean(yf),np.mean(xf)])
    averages=np.mean(totals,axis=0)
    assert averages[0]>0 and averages[1]>0
    measurements[law]=averages.tolist()
testseeds=[]
for l in range(len(LAWS)):
    for b in range(d['blocks_per_law']):
        rng=np.random.default_rng(np.random.SeedSequence([30091249,l,b]))
        rng.permutation(8); seeds=rng.integers(1000,2**30,size=11)
        testseeds.extend(seeds[:10])
previous=json.loads((V29/'design.json').read_text())
historic={e['seed'] for e in previous['environments']} | {i for p in previous['bank_pairs'] for i in p}
assert len(set(testseeds))==len(testseeds) and not (historic & set(testseeds))
assert np.array_equal(quota_actions(np.zeros((8,128,3)))[:,:13],np.full((8,13,3),3))
out=dict(training_shape=list(rows.shape),label_shape=list(truth.shape),laws=measurements,
         description="Diagnostic seeds only, distinct from test. Mean [stick flip rate, probe change, factual change]. No fitting or inference.",
         disjoint_seeds=len(testseeds))
atomic_json(ROOT/'preflight.json',out)
print(json.dumps(out))
