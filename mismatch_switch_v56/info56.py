"""Independent, outcome-blind pilot ceilings and counterfactual population checks."""
import json
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
W = np.array([.35, .52])
THRESHOLD = .4348074444131813
MIX = np.column_stack(((1-W)/2, (1-W)/2, W))
ALIAS = np.array([[.42,0,.58],[0,.84,.16]])

def pilot_likelihood(k, active):
    """Probability of each (usable, flips) count in the two equally likely populations."""
    rows, action = [], []
    mismatch = .125 + .375*W
    for n in range(k+1):
        usable = comb(k,n)*active**n*(1-active)**(k-n)
        for flips in range(n+1):
            rows.append(usable*comb(n,flips)*mismatch**flips*(1-mismatch)**(n-flips))
            action.append(n>0 and flips > (.125+.375*THRESHOLD)*n)
    prob = np.asarray(rows)
    np.testing.assert_allclose(prob.sum(0),1)
    return prob,np.asarray(action)

def ideal(k, active, gap):
    # We subtract fixed-impact loss separately in each stratum, leaving costs [gap,0].
    prob, use_fixed=pilot_likelihood(k,active)
    risk_by_count=prob@(gap*.5)
    best_constant=min(0.,gap.mean())
    minimum=np.minimum(risk_by_count,0.).sum()
    perfect=np.minimum(gap,0.).mean()
    return dict(K=k, usable_probability=active,
                max_balanced_accuracy=.5*prob.max(axis=1).sum(),
                predicted_fixed_rate_by_w=prob[use_fixed].sum(axis=0).tolist(),
                best_possible_gain_over_best_constant=best_constant-minimum,
                frozen_threshold_gain_over_best_constant=best_constant-risk_by_count[~use_fixed].sum(),
                perfect_w_gain_over_best_constant=best_constant-perfect)

def analyze(by_e):
    """Input is 24 x 3 catalog means of (frozen v54 gate Brier minus fixed I)."""
    assert by_e.shape==(24,3)
    gap=by_e.mean(axis=0)
    scenario=MIX@gap
    opts=ALIAS@gap
    expect=ALIAS@np.array([0,.25,.5])
    np.testing.assert_allclose(expect,[.29,.29])
    boot=np.random.default_rng(560211).integers(len(by_e),size=(10000,len(by_e)))
    dist=by_e[boot].mean(axis=1)@ALIAS.T
    doc=dict(mean_by_e=gap.tolist(),primary_gap=scenario.tolist(),alias_mean=opts.tolist(),
             alias_paired_ci=np.quantile(dist,[.0125,.9875],axis=0).T.tolist(),
             alias_mismatch=expect.tolist(),
             alias_two_flips=(ALIAS@(np.array([0,.25,.5])**2)).tolist(),
             ideal=[ideal(k,ac,scenario) for ac in (.5,1.) for k in (0,1,4,8,16,32,64)])
    (ROOT/'identification56.json').write_text(json.dumps(doc,indent=2)+'\n')

if __name__=='__main__':
    with np.load(ROOT/'score56_cells.npz') as f:
        costs=f['by_e'].mean(axis=1)  # 24 catalogs, three e levels, gate and fixed.
        analyze(costs[...,0]-costs[...,1])
        gaps=f['by_e'][...,0]-f['by_e'][...,1]
    endpoints=np.stack((gaps[:,:,:2].mean(axis=2),gaps[:,:,2]),axis=-1)
    a,b=endpoints.mean((0,1))
    mask=endpoints[:,:,0]*endpoints[:,:,1]<0
    crossing=-endpoints[:,:,0][mask]/(endpoints[:,:,1]-endpoints[:,:,0])[mask]
    dest=ROOT/'identification56.json'
    doc=json.loads(dest.read_text())
    doc['fresh_population_crossing']=float(-a/(b-a))
    doc['fresh_materials_with_crossing']=int(mask.sum())
    doc['fresh_material_crossing_quantiles']=np.quantile(crossing,[0,.25,.5,.75,1]).tolist() if mask.any() else []
    dest.write_text(json.dumps(doc,indent=2)+'\n')
