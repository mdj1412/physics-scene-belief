"""Independent sign-power diagnostic: true alpha_c is supplied and only calibration sensors are observed."""
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'population_shift_v27'))
from estimator28 import ProbeShiftEstimator,factual_mean,PROBE_INCOMING,sigmas,t,torch
from runner27 import draw_theta,stream,impact

BUDGETS=(0,1,4,16)
COHORTS=256
GRID=np.linspace(-1.,1.,401)
Q=np.ones(len(GRID))
Q[[0,-1]]=.5

def probabilities(log_ratios):
    # Scaling per row removes common constants. Use logaddexp for endpoint stability.
    with np.errstate(divide='ignore'):
        lw=np.log((1+GRID)/2)
        lf=np.log((1-GRID)/2)
    ll=np.logaddexp(log_ratios[:,:,None]+lw,lf)
    profile=ll.sum(axis=1)
    posterior=np.exp(profile-profile.max(1,keepdims=True))*Q
    posterior/=posterior.sum(1,keepdims=True)
    neg=posterior[:,GRID<0].sum(1)+.5*posterior[:,GRID==0].ravel()
    return neg,posterior@GRID

def latent_oracle(k,a):
    p=(1+abs(a))/2
    return float(sum(math.comb(k,j)*p**j*(1-p)**(k-j)*((j>k/2)+.5*(j==k/2)) for j in range(k+1)))

def wilson(n,success):
    frac=success/n
    z=1.95996398454
    denom=1+z*z/n
    c=(frac+z*z/(2*n))/denom
    w=z/denom*math.sqrt(frac*(1-frac)/n+z*z/(4*n*n))
    return [max(0.,c-w),min(1.,c+w)]

def measure(neg,estimate,af,k):
    n=len(neg)
    conf=(neg>.95)|(neg<.05)
    # Exact k=0 posterior is symmetric, sign abstains and receives half credit.
    correct=neg if af<0 else 1-neg
    ties=(np.abs(neg-.5)<1.e-12)
    signs=(correct>.5+1.e-12).astype(float)+.5*ties
    if af==0:
        accuracy=ci=confidence_errors=post_correct=None
    else:
        accuracy=float(signs.mean())
        ci=wilson(n,signs.sum()) if k else [.5,.5]
        confidence_errors=int((conf&(correct<.5)).sum())
        post_correct=float(correct.mean())
    return dict(mean_alpha_hat=float(estimate.mean()),mae=float(np.mean(abs(estimate-af))),
                sign_accuracy=accuracy,accuracy_mc95=ci,posterior_correct=post_correct,
                n_confident=int(conf.sum()),n_confident_wrong=confidence_errors,
                confident_fraction=float(conf.mean()),confidence_mc95=wilson(n,conf.sum()),
                sign_tie_fraction=float(ties.mean()))

def ratio_fixed_c(log_components,ac):
    log_components=log_components.reshape(-1,2,2)
    with np.errstate(divide='ignore'):
        p0=np.log((1+ac)/2)
        p1=np.log((1-ac)/2)
    factors=np.logaddexp(log_components[:,0]+p0,log_components[:,1]+p1)
    return (factors[:,0]-factors[:,1]).reshape(COHORTS,max(BUDGETS))

@torch.inference_mode()
def run():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('BELIEF_DEVICE')=='cpu' or torch.cuda.device_count()!=1:
        raise RuntimeError('Use one idle allocated GPU.')
    torch.set_num_threads(2)
    design=json.loads((ROOT/'design.json').read_text())
    laws=[e for e in design['environments'] if e['rep']==0 and (e['target'] or e['family']=='null')]
    assert len(laws)==9
    seed=28278151
    bank_seed=28297361
    estimator=ProbeShiftEstimator(seed=bank_seed,bank_size=8192)
    pcenters=estimator.joint_centers[:,:,-2:]
    pnorm=.5*pcenters.square().sum(-1)
    rows=[]
    for j,e in enumerate(laws):
        ac,af=e['alpha']
        case_seed=seed+13007*j
        n=COHORTS*max(BUDGETS)
        theta=draw_theta(n,case_seed,10,(ac,af))
        x=factual_mean(theta)+t(stream(case_seed,51).normal(size=(n,11)))*sigmas()[0]
        probe,sticks=impact(theta,0,PROBE_INCOMING)
        assert not bool(sticks.any())
        yy=probe+t(stream(case_seed,52).normal(size=(n,2)))*sigmas()[3]
        full=estimator.joint_component_loglikelihood(x,yy)
        # Probe-only includes both observed outgoing components of the calibration response.
        only=[]
        for yblock in yy.split(128):
            particle=(yblock/sigmas()[3])@pcenters.transpose(1,2)-pnorm[:,None,:]
            only.append((torch.logsumexp(particle,dim=-1).T-math.log(estimator.bank_size)).cpu().numpy())
        ratios={'factual_probe':ratio_fixed_c(full,ac),'probe_only':ratio_fixed_c(np.concatenate(only),ac)}
        for k in BUDGETS:
            methods={}
            for method,data in ratios.items():
                neg,hat=probabilities(data[:,:k])
                methods[method]=measure(neg,hat,af,k)
            rows.append(dict(family=e['family'],alpha_c=ac,alpha_f=af,n_probes=k,cohorts=COHORTS,
                         methods=methods,latent_bit_oracle_accuracy=latent_oracle(k,af) if af else None))
        print('calibration sign power',e['family'],flush=True)
    out=dict(scope='Known profile strength; only calibration x and probe y, no decision metric. Independent from primary.',
        cohorts_per_law=COHORTS,budgets=BUDGETS,bank_seed=bank_seed,data_seed=seed,
        positive_directional_call='P(alpha_f<0|data)>0.95 or <0.05',grid_points=len(GRID),
        null_confidence='All confident sign calls are false directional calls when alpha_f=0.',
        oracle='Exact equal-prior sign accuracy given known magnitude and K perfect latent flip bits.',
        uncertainty='Pointwise 95% Wilson Monte Carlo intervals, per law; K=0 analytic tie/abstention.',
        device=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],rows=rows)
    path=ROOT/'results'/'power28.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(path)

if __name__=='__main__':run()
