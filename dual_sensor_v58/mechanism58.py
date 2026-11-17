"""Mechanism identification from one blind pilot, plus an exact common-mode counterexample."""
from itertools import product
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
Q = 0.29
H = (.145 - .0925) / (.29 - .0925)
D = 2 * .02 * .98
PARAMS = list(product(range(2), repeat=3))  # regime, physical persistence, latch
EPS = 1e-10


def binary_pair(mean, joint):
    return np.array([[1 - 2 * mean + joint, mean - joint], [mean - joint, joint]])


def pair_emission(layout, regime, persistence, latch):
    share = layout == 1
    joint = [.145, .0925][regime] if share and persistence else Q ** 2
    p_truth = binary_pair(Q, joint)
    table = np.zeros((5, 5))
    # For a single sensor, marginalize latched phantom and independent conversion errors.
    for x1, x2 in product(range(2), repeat=2):
        latent = np.zeros((2, 2))
        latent[x1, x2] = 1
        latched = latch * H
        if share:
            before = (1 - latched) * latent + latched * np.diag([1-Q, Q])
        else:
            first, second = (latched * Q + (1-latched) * x for x in (x1, x2))
            before = np.outer([1-first, first], [1-second, second])
        crossover = np.array([[1-D,D],[D,1-D]])
        reads = crossover.T @ before @ crossover
        for a,b,c,d in product(range(2),repeat=4):
            s1,s2 = 1+a+2*c, 1+b+2*d
            table[s1,s2] += p_truth[x1,x2] * reads[a,b] * reads[c,d]
    table[0,1:] = table[1:,1:].sum(axis=0)
    table[1:,0] = table[1:,1:].sum(axis=1)
    table[0,0] = 1
    table *= .25  # equal probability of each combination of active flags
    assert np.isclose(table.sum(), 1.0)
    return table.ravel()


def coupling():
    # Device and physical worlds share apparent signs but have distinct latent ground truths.
    rng = np.random.default_rng(58061701)
    c = (.145-Q**2)/(Q*(1-Q))
    truth = rng.random((262144,2)) < Q
    latch = rng.random((len(truth),1)) < c
    z = rng.random((len(truth),1)) < Q
    physical = np.where(latch,z,truth)
    # Recover a genuine A session state using its exact conditional law.
    ones = physical.sum(axis=1)
    p_high = np.where(ones==0, .58*.25/(.42+.58*.25), 1.)
    high = rng.random(len(truth)) < p_high
    active = rng.random(truth.shape) < .5
    eps = rng.random((*truth.shape,2)) < D

    def observe(visible_truth):
        both = visible_truth[:,:,None] ^ eps
        return active * (1 + both[:,:,0].astype(int) + 2*both[:,:,1].astype(int))
    left,right = observe(physical),observe(np.where(latch,z,truth))
    p_phys = binary_pair(Q,.145)
    p_app = (1-c)*binary_pair(Q,Q**2) + c*np.diag([1-Q,Q])
    assert np.allclose(p_phys,p_app) and np.array_equal(left,right)
    return dict(common_latch=c,max_probability_error=float(abs(p_phys-p_app).max()),
                paired_groups=len(truth),different_observed_symbols=int((left!=right).sum()),
                true_double_physical=physical.all(axis=1).mean(),true_double_artifact=truth.all(axis=1).mean(),
                apparent_double=physical.all(axis=1).mean(),physical_high_session_fraction=high.mean())


def evaluate():
    indices = np.asarray(PARAMS)
    tables = np.array([[pair_emission(layout,*p) for p in PARAMS] for layout in (0,1)])
    assert np.max(np.abs(tables[0,0]-tables[0,2])) < 1e-12
    from route58 import TABLE
    a,b=np.divmod(np.arange(25),5)
    grouped=np.exp(TABLE.reshape(8,25))*.25
    single=np.exp((TABLE[...,5*a]+TABLE[...,b]).reshape(8,25))*.25
    agreement=float(np.max(abs(tables-np.stack([single,grouped]))))
    assert agreement < 1e-12, agreement
    logs = np.log(tables)
    files = sorted((ROOT/'pilots').glob('panel_*.npz'))
    assert len(files)==24
    cm = np.zeros((24,2,4,4),dtype=np.int64)
    hits = np.zeros((24,2,2,2))  # catalog, layout, target, truth -> P(class=1)
    for ic, file in enumerate(files):
        with np.load(file) as archive:
            codes = archive['codes']
        assert codes.shape[:5]==(2,2,2,2,2) and codes.shape[-1]==4
        for layout,rho,h in product(range(2),repeat=3):
            rows = codes[layout,rho,h].reshape(-1,4)
            ll = logs[layout][:, rows].sum(axis=2)
            w = np.exp(ll - ll.max(axis=0))
            wp = w.reshape(2,4,-1).sum(axis=0)
            # Make no-effect ties deterministic, independent of floating-point order.
            pred = (wp >= wp.max(axis=0)-EPS).argmax(axis=0)
            cm[ic,layout,2*rho+h] = np.bincount(pred,minlength=4)
            for k,true in enumerate((rho,h)):
                diff=w[indices[:,k+1]==1].sum(axis=0)-w[indices[:,k+1]==0].sum(axis=0)
                hits[ic,layout,k,true] += (diff>EPS).mean()/2
    joint=cm/np.sum(cm,axis=-1,keepdims=True)
    sensitivity = hits[:,:,:,1]
    specificity = 1-hits[:,:,:,0]
    ba=(sensitivity+specificity)/2
    rates=np.concatenate([ba,joint.diagonal(axis1=-2,axis2=-1).mean(axis=-1)[:,:,None]],axis=-1)
    rng=np.random.default_rng(58061702)
    boot=rng.integers(0,len(files),(10000,len(files)))
    deltas=(rates[:,1]-rates[:,0])[boot].mean(axis=1)
    return dict(likelihood_implementation_error=agreement,order=['persistence','latch','joint_four_cell_MAP'], balanced_or_joint_accuracy=rates.mean(axis=0).tolist(),
                delta_4x2_minus_8x1=(rates[:,1]-rates[:,0]).mean(axis=0).tolist(),
                delta_pointwise95=np.quantile(deltas,[.025,.975],axis=0).tolist(),
                sensitivity=sensitivity.mean(axis=0).tolist(),specificity=specificity.mean(axis=0).tolist(),
                joint_confusion_layout_true_pred=joint.mean(axis=0).tolist(),
                layout=['8x1','4x2'],cause_order=['neither','latch only','persistence only','both'],
                coupling=coupling())


if __name__=='__main__':
    results=evaluate()
    (ROOT/'mechanism58.json').write_text(json.dumps(results,indent=2)+'\n')
