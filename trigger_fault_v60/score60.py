"""Evaluate the frozen policy on paired markers, delays, hardware fault distributions and fresh deployment outcomes."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'sensor_reset_v59'))
from route59 import SOURCE_LOG_CUTOFF, log_odds
from score59 import MIX, EVENT, cdf_boot, risk_by_catalog, weights

# A/B testing regimes and P/C mechanism are separate axes. Pairing order is fixed.
ARMS = [(1,1), (1,0), (0,1), (0,0)]  # sync,reset
NAMES = ['risk_sync_reset','risk_sync_sham','risk_raw_reset','risk_raw_sham',
         'map_sync_reset','map_sync_sham','map_raw_reset','map_raw_sham','gate','fixed','oracle']
TARGETS = ['matched','independent','reversed','common_high','common_low']
DELAY = [0,.2,1]
C = 24

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pilot_report():
    rates=[]; causes=[]; gate=[]; fine=[]; ultra=[]; real=[]; meta=[]
    for c in range(C):
        pilot=ROOT/'pilots'/f'panel_{c:02d}.npz'
        forecast=ROOT/'scored60'/f'panel_{c:02d}.npz'
        with np.load(pilot) as f:
            codes=f['codes']
        assert codes.shape[:6] == (3,2,2,2,2,2) and codes.shape[-1]==4 and codes.max()<169
        n=codes.shape[-2]
        risk=[]; map_action=[]; diagnosis=[]
        for sync,reset in ARMS:
            p,m=log_odds(codes[:,sync,reset],reset,True)
            risk.append((p>SOURCE_LOG_CUTOFF).mean(axis=-1))
            map_action.append((p>1e-12).mean(axis=-1))
            diagnosis.append((m>1e-12).mean(axis=-1))
        zeros=np.zeros_like(risk[0]) # d, mechanism, env, material
        oracle=np.broadcast_to(np.array([1.,0.])[None,None,:,None],zeros.shape)
        rates.append(np.stack([*risk,*map_action,zeros,1-zeros,oracle],axis=-1).transpose(0,3,1,2,4))
        causes.append(np.stack(diagnosis,axis=-1).transpose(0,3,1,2,4))
        material=((np.arange(4)+c)%4)//2
        with np.load(forecast) as f:
            for dest,key in ((fine,'exact_cells'),(ultra,'ultra_cells')):
                dest.append(np.stack([f[key][:,:,material==j,:2].mean((1,2)) for j in (0,1)]))
            gate.append([f['gate'][:,material==j].mean() for j in (0,1)])
            x=f['real'][...,:2]
            assert x.shape==(2,3,2,5,4,256,2)
            real.append(np.stack([np.take(x,np.flatnonzero(material==j),axis=4).reshape(2,3,2,-1,2)
                                  for j in (0,1)],axis=1))
        meta.append([digest(pilot),digest(forecast)])
    return list(map(np.asarray,(rates,causes,gate,fine,ultra,real,meta))),n

def assign(arr,key):
    """Assign common hazards to regime at evaluation; keep the original posterior and fixed actions."""
    if key=='matched':
        return arr
    buf=arr.copy() # C,D,F,K,Z,M
    common=arr[:,:,:,1]
    if key=='independent':
        new=common.mean(axis=-2,keepdims=True)
    elif key=='reversed':
        new=common[...,::-1,:]
    elif key=='common_low':
        new=common[...,1:2,:]
    elif key=='common_high':
        new=common[...,:1,:]
    else:
        raise KeyError(key)
    buf[:,:,:,1]=new
    return buf

def cat_metrics(fine,rate,causes):
    # Inputs have C,F,K,Z,M, C,F,K,Z,arms
    action=rate.mean(2)
    risk = risk_by_catalog(fine,action).mean((1,2))
    acc=(action[:,:,0]+1-action[:,:,1]).mean(1)/2
    cause_acc=((causes[:,:,0]+1-causes[:,:,1])/2).mean((1,2))
    return {'mean_Brier':risk,'action_BA':acc,'cause_BA':cause_acc}

def compare(x,a,b,n):
    diff=x[:,a]-x[:,b]
    return [diff[0],*np.quantile(diff[1:],[.025/n,1-.025/n])]

def pack(fine,rate,causes,raw,boot,primary=False,adjust=18):
    # Full bootstrap for mean and action/diagnosis. Session tail CI for confirmatory population.
    cats=cat_metrics(fine,rate,causes)
    draws={k:boot@v for k,v in cats.items()}
    r=rate.mean(2)
    target_losses=np.einsum('cfea,ze->cfza',fine,MIX)
    optimum=target_losses[...,1]<target_losses[...,0]
    pos=optimum.sum((1,2))
    neg=(~optimum).sum((1,2))
    # Rare violation of the A/B mapping: calculate true lower-Brier action balanced accuracy.
    tp=(optimum[...,None]*r).sum((1,2))
    tn=((~optimum)[...,None]*(1-r)).sum((1,2))
    draws['lower_risk_action_BA']=.5*((boot@tp)/(boot@pos)[:,None]+(boot@tn)/(boot@neg)[:,None])
    ws=weights(r)
    tail=cdf_boot(raw[0],ws, boot if primary else boot[:1])
    draws['session_p90']=tail
    out={k:v[0] for k,v in draws.items()}
    out['primary_vs_sham_gate']={}
    for k,m in draws.items():
        if k=='cause_BA':
            out['primary_vs_sham_gate'][k]=[compare(m,0,1,2),None]
        elif k!='session_p90' or primary:
            correction=2 if primary else (adjust if k=='mean_Brier' else 1)
            out['primary_vs_sham_gate'][k]=[compare(m,0,j,correction) for j in [1,8]]
        else:
            out['primary_vs_sham_gate'][k]=[[m[0,0]-m[0,j]] for j in [1,8]]
    out['cause_BA_by_env']=((causes[:,:,0]+1-causes[:,:,1])/2).mean((0,1))
    out['action_BA_by_cause']=((rate[:,:,:,0]+1-rate[:,:,:,1])/2).mean((0,1))
    out['fixed_rate_A_B']=r.mean((0,1))
    out['fixed_rate_by_cause_env']=rate.mean((0,1))
    if primary:
        out['diagnostics']={}
        for key,matrix in draws.items():
            pairs=[(0,2),(1,3),(0,1)] if key=='cause_BA' else [(0,2),(1,3),(4,5),(4,8),(0,9)]
            out['diagnostics'][key]=[compare(matrix,i,j,1) for i,j in pairs]
        out['seed2_p90']=cdf_boot(raw[1],ws,boot[:1])[0]
        # Directly check that F(x^-) < .90 <= F(x), independently of grouped inverse CDF.
        for k in (0,1,8):
            at=tail[0,k]
            w=ws[k][...,None]/(C*raw.shape[-1])
            assert (w*(raw[0]<at)).sum()<=.90+1e-8 and (w*(raw[0]<=at)).sum()>=.90-1e-8
    return out

def run():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for row in (ROOT/'score_freeze60.sha256').read_text().splitlines():
        sha,name=row.split(maxsplit=1)
        assert digest(ROOT/name)==sha
    (rates,causes,gate,fine,ultra,real,hashes),n=pilot_report()
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    replay=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    rng=np.random.default_rng(60091103)
    cat=rng.multinomial(C,np.ones(C)/C,size=10000)/C
    w=np.vstack([np.ones((1,C))/C,cat])
    report={'methods':NAMES,'arms_sync_reset':ARMS,'delays':DELAY,'episodes_per_material_cell':n,'cells':{},'uniform_delay':{}}
    for target in TARGETS:
        a=assign(rates,target);b=assign(causes,target)
        # A privileged target-regime action knows the deployment state and ignores the C-pilot assignment.
        a[...,-1]=rates[...,-1]
        report['cells'][target]=[pack(fine,a[:,d],b[:,d],raw,w,adjust=18 if target in TARGETS[:3] else 1) for d in range(3)]
        is_primary=target=='independent'
        v=a.mean(1);bc=b.mean(1)
        result=pack(fine,v,bc,raw,w,primary=is_primary,adjust=1)
        result['deployment_final_impact_rate']=(gate[:,:,None,None]+(1-gate[:,:,None,None])*v.mean(2)).mean((0,1,2))
        if is_primary:
            checks={}
            for label,source in zip(['ultra','binary_seed0','binary_seed1'],[ultra,*replay]):
                x=w@risk_by_catalog(source,v.mean(2)).mean((1,2))
                checks[label]={'mean':x[0],'vs_sham_gate':[compare(x,0,j,2) for j in (1,8)]}
            result['robustness']=checks
        report['uniform_delay'][target]=result
    target_brier=np.einsum('cfea,ze->cfza',fine,MIX)
    gap=target_brier[...,0]-target_brier[...,1]
    report['lower_risk_mean_privileged_oracle']=target_brier.min(axis=-1).mean();report['A_mapping_violations']=[dict(catalog=int(c),material=int(m),gate_minus_fixed=float(gap[c,m,0])) for c,m in np.argwhere(gap[...,0]<=0)];report['gate_minus_fixed_A_B']=gap.mean((0,1));report['correct_mapping_A_B']=[(gap[...,0]>0).sum(),(gap[...,1]<0).sum()]
    np.savez_compressed(ROOT/'score60_cells.npz',rates=rates,cause=causes,fine=fine,ultra=ultra,replay=replay,
                        gates=gate,input_sha256=hashes,methods=NAMES,delay=DELAY)
    (ROOT/'score60.json').write_text(json.dumps(report,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    print('finished',report['uniform_delay']['independent']['primary_vs_sham_gate'])
if __name__=='__main__':
    run()
