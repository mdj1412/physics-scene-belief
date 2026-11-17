"""Rare paired purchase discordance: independent and exchangeable-surface planning."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import sys
import numpy as np
import scipy
from scipy.stats import beta, betabinom, binom
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "reflective_field_v64"
LEGACY = ROOT.parent / "reflective_tracking_v63"
SEED = 6503172

def recover_v64():
    sys.path.insert(0, str(LEGACY))
    import score63
    by_surface = []
    for c in range(24):
        with np.load(SOURCE/'pilots'/f'panel_{c:02d}.npz') as f:
            arms = score63.split_codes(f['codes'][:,2,1])
            choices = score63.classify(arms.reshape(-1,8))[:,0].reshape(arms.shape[:-1])
        with np.load(LEGACY/'scored63'/f'panel_{c:02d}.npz') as f:
            gate = f['gate']
        material = ((np.arange(4) + c) % 4) // 2
        flip = choices[::2] != choices[1::2]
        for m in (0,1):
            rate = float(flip[:,:,:,m,:].mean())
            gate_r = float((~gate[:,material==m]).mean())
            by_surface.append(dict(catalog=c, material=m, router_flip=rate,
                                    gate_R=gate_r, terminal_flip=rate*gate_r))
    observed = json.loads((SOURCE/'sim_score64.json').read_text())['lamp_switch_uniform_width']
    np.testing.assert_allclose(np.mean([x['terminal_flip'] for x in by_surface]), observed[1][0], atol=1e-15)
    return by_surface, observed[1]

def distribution(p, icc, m, completion=1.):
    j = np.arange(m+1)
    if icc==0 or m==1: return binom.pmf(j,m,p*completion)
    precision = 1/icc-1
    counts = betabinom.pmf(j,m,p*precision,(1-p)*precision)
    if completion<1:
        counts = sum(freq * np.nan_to_num(binom.pmf(j, n, completion))
                     for n,freq in enumerate(counts))
    counts /= counts.sum()
    np.testing.assert_allclose(counts@j, p*m*completion, atol=1e-10)
    return counts

def at_least(pmf, surfaces, threshold):
    # Truncation is exact for the probability of a total below threshold.
    if surfaces == 0: return 0.
    a = pmf[:threshold]
    out = np.array([1.])
    while surfaces:
        if surfaces%2: out = np.convolve(out,a)[:threshold]
        a = np.convolve(a,a)[:threshold]
        surfaces //= 2
    return float(np.clip(1-out.sum(),0,1))

def first(threshold, probability):
    upper=1
    while probability(upper)<threshold:
        upper*=2
    lower=0
    while upper-lower>1:
        mid=(upper+lower)//2
        if probability(mid)>=threshold: upper=mid
        else: lower=mid
    assert probability(upper-1)<threshold<=probability(upper)
    return upper

def design(p,icc,m,completion=1.):
    pmf=distribution(p,icc,m,completion)
    q = 1-pmf[0]
    event=lambda s: at_least(pmf,s,10)
    one=lambda s: -np.expm1(s*np.log1p(-q))
    distinct=lambda s: binom.sf(9,s,q)
    return {'p':p,'icc':icc,'sessions_per_surface':m,'complete_chance':completion,
            'one_event_95':first(.95,one),'ten_events_90':first(.9,event),
            'ten_event_surfaces_90':first(.9,distinct),
            'RSE_25':math.ceil((1-p)*(1+(m-1)*icc)/(p*.25**2*m)),
            'RSE_15':math.ceil((1-p)*(1+(m-1)*icc)/(p*.15**2*m))} if completion==1 else {
            'one_event_95':first(.95,one),'ten_events_90':first(.9,event),
            'ten_event_surfaces_90':first(.9,distinct)}

def precision_iid(p,fraction):
    # CP width is symmetric in k and increases up to the midpoint.
    n = np.arange(100,500001,100)
    k = binom.ppf(.90,n,p).astype(int)
    lower = np.zeros_like(n,dtype=float)
    lower[k>0] = beta.ppf(.025,k[k>0],n[k>0]-k[k>0]+1)
    upper = beta.ppf(.975,k+1,n-k)
    halfwidth = (upper-lower)/2
    idx = np.flatnonzero(halfwidth<=p*fraction)[0]
    ni=int(n[idx])
    # Sum the exact binomial probability of all widths meeting the target.
    limit=int(binom.ppf(1-1.e-14,ni,p)) + 10
    x = np.arange(limit+1)
    lo = np.r_[0.,beta.ppf(.025,x[1:],ni-x[1:]+1)]
    hi = beta.ppf(.975,x+1,ni-x)
    success = ((hi-lo)/2<=p*fraction)
    assurance=float(binom.pmf(x,ni,p)[success].sum())
    assert assurance>=.9-1.e-12
    return dict(N_grid_100=ni, expected_events=ni*p, width_target=2*fraction*p,
                assurance=assurance,tail_omitted=binom.sf(limit,ni,p))

def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    surf, v64=recover_v64()
    p=v64[0]; scen={'reference':p,'low':v64[1],'high':v64[2]}
    all_rows = []
    for tag,prob in scen.items():
        for m in (1,4,8,16,32):
            for icc in (0.,.01,.05,.10):
                row=design(prob,icc,m)
                row['scenario']=tag
                all_rows.append(row)
    small=[]
    for s,m in [(12,4),(16,4),(24,8),(48,4),(48,8),(24,16),(96,4)]:
        for tag,prob in scen.items():
            for icc in (0.,.05,.10):
                pmf=distribution(prob,icc,m)
                prob_zero=lambda pp:float(distribution(pp,icc,m)[0]**s)
                u=brentq(lambda pp:prob_zero(pp)-.05,1e-9,.99)
                small.append(dict(surfaces=s,sessions_per_surface=m,p=prob,scenario=tag,icc=icc,
                    expected_events=s*m*prob,probability_zero=prob_zero(prob),probability_ten=at_least(pmf,s,10),
                    zero_event_one_sided_95_model_upper=u))
    independent = {tag: {str(frac):precision_iid(prob,frac) for frac in (.5,.25)} for tag,prob in scen.items()}
    check = []
    rng = np.random.default_rng(SEED)
    for m,s,icc in [(4,12,.05),(8,230,.05),(8,1000,.05),(8,1200,.1),(8,950,0)]:
        pmf=distribution(p,icc,m)
        draws=rng.multinomial(s, pmf, size=100000)
        total=draws @ np.arange(m+1)
        pos=s-draws[:,0]
        quantities=[(total>=1,at_least(pmf,s,1)),(total>=10,at_least(pmf,s,10)),
                    (pos>=10,float(binom.sf(9,s,1-pmf[0])))]
        estimates=[]
        for outcomes,analytic in quantities:
            estimated=float(outcomes.mean()); se=float(np.sqrt(analytic*(1-analytic)/len(outcomes)))
            assert abs(estimated-analytic)<6*se+1e-8
            estimates.append([estimated,analytic,se])
        check.append(dict(m=m,s=s,icc=icc,mc_and_exact_and_mc_se=estimates))
    for m,s in [(1,16),(8,231),(32,1000)]:
        for k in (1,10):
            np.testing.assert_allclose(at_least(distribution(p,0,m),s,k),binom.sf(k-1,m*s,p),atol=2e-12)
    out=dict(v64_terminal_rate=v64, surface_replay=surf, designs=all_rows,
             scenarios=scen, small_designs=small, iid_precision=independent, monte_carlo=check,
             recruitment_90pct=design(p,.05,8,.9),monte_carlo_seed=SEED,mc_reps=100000,
             versions={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                       'host':platform.node(),'slurm':os.getenv('SLURM_JOB_ID')},
             sha256_script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'power65.json').write_text(json.dumps(out,indent=2)+'\n')
    print('validated', len(all_rows),'designs, three targets and five Monte Carlo sets.')
if __name__=='__main__':main()
