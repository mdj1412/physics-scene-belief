"""Exact eight-attempt likelihood and Brier decision-value diagnostics."""
from itertools import product
from math import comb
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EPS = np.array([0., .25, .5])
MIX = np.array([[.42,0,.58],[0,.84,.16]])
Q = .29

def one_session(group_size, active, rho):
    states, prob = [], []
    for n in range(group_size+1):
        valid_prob = comb(group_size,n)*active**n*(1-active)**(group_size-n)
        for x in range(n+1):
            shared = MIX @ (EPS**x*(1-EPS)**(n-x))
            independent = Q**x*(1-Q)**(n-x)
            prob.append(valid_prob*comb(n,x)*(rho*shared+(1-rho)*independent))
            states.append((n,x))
    return np.array(states), np.array(prob)

def table(group_size,active,rho):
    states,p=one_session(group_size,active,rho)
    ids=np.array(list(product(range(len(states)),repeat=8//group_size)))
    mass=np.prod(p[ids],axis=1)
    np.testing.assert_allclose(mass.sum(0),[1,1])
    full_groups=(states[ids,0]>=2).sum(1)
    return mass,full_groups

def summary(group_size,active,rho,source_gap,target_gap):
    mass,full_groups = table(group_size,active,rho)
    model,_ = table(group_size,active,1.)
    eps=1e-12
    # Ties use B's fixed action (the v54 gate).
    use_i = model[:,0] > model[:,1]*(1+eps)
    cost_sensitive=model[:,0]*source_gap[0] > -model[:,1]*source_gap[1]*(1+eps)
    acc=.5*np.maximum(mass[:,0],mass[:,1]).sum()
    def evaluate(use_fixed):
        rates=mass[use_fixed].sum(0)
        return dict(fixed_rate_by_type=rates.tolist(),
                    balanced_accuracy=float((rates[0]+1-rates[1])/2),
                    gain_vs_gate=float(rates@target_gap/2))
    gains=mass@target_gap/2
    return dict(group_size=group_size,active=active,rho=rho,max_accuracy=float(acc),
                posterior_MAP=evaluate(use_i),source_risk=evaluate(cost_sensitive),
                max_gain_vs_gate=float(np.maximum(gains,0).sum()),
                p_no_complete_group=float(mass[full_groups==0,0].sum()),
                p_2plus_complete_groups=float(mass[full_groups>=2,0].sum()),
                gain_perfect_type_vs_gate=float(np.maximum(target_gap,0).mean()))

def main():
    src=json.loads((ROOT.parent/'mismatch_switch_v56'/'identification56.json').read_text())
    source_gap=np.asarray(src['alias_mean'])
    with np.load(ROOT/'score57_cells.npz') as data:
        base=data['by_e'].mean(axis=1)
    gaps=base[...,0]-base[...,1] # catalog × e, two common decoder risks
    assert gaps.shape==(24,3)
    target_gap=MIX@gaps.mean(0)
    mu=MIX@EPS
    m2=MIX@EPS**2
    np.testing.assert_allclose(mu,Q)
    h=(m2[0]-m2[1])/(mu[1]-m2[1])
    np.testing.assert_allclose(h*Q+(1-h)*m2[1],m2[0])
    partial=[summary(2,.5,rho,source_gap,target_gap) for rho in (0,.25,.5,.75,1.)]
    layouts=[summary(size,active,1,source_gap,target_gap) for active in (.5,1.) for size in (1,2,4)]
    out=dict(layouts=layouts,partial=partial,marginal=mu.tolist(),second=m2.tolist(),
             copy_probability_B_to_match_A=float(h),source_gap_A_B=source_gap.tolist(),fresh_gap_A_B=target_gap.tolist())
    (ROOT/'information57.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
