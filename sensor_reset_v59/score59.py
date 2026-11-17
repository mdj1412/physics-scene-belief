"""Evaluate independently generated pilot decisions on fresh simulated deployment outcomes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from route59 import METHODS, ARMS, log_odds, SOURCE_LOG_CUTOFF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'mismatch_switch_v56'))
from score56 import cdf_boot, EVENT
MIX = np.array([[.42,0,.58],[0,.84,.16]])
C = 24
COMPARISONS = [(0,1),(2,3),(0,2),(1,3)]


def read_panels():
    fine,ultra,real,rates,gates,digests,causes=[],[],[],[],[],[],[]
    for c in range(C):
        m=((np.arange(4)+c)%4)//2
        score_file=ROOT/'scored59'/f'panel_{c:02}.npz'
        with np.load(score_file) as f:
            for target,key in ((fine,'exact_cells'),(ultra,'ultra_cells')):
                target.append(np.stack([f[key][:,:,m==g,:2].mean((1,2)) for g in range(2)]))
            gates.append([f['gate'][:,m==g].mean() for g in range(2)])
            x=f['real'][...,:2]
            assert x.shape==(2,3,2,5,4,256,2)
            real.append(np.stack([np.take(x,np.flatnonzero(m==g),axis=4).reshape(2,3,2,-1,2) for g in (0,1)],axis=1))
        pilot_file=ROOT/'pilots'/f'panel_{c:02}.npz'
        with np.load(pilot_file) as f:
            codes=f['codes']
        assert codes.shape==(2,2,2,2,2,2048,4) and codes.max()<169
        map_actions=[];risk_actions=[];cause_rates=[]
        for arm in ARMS:
            reset, camera=int(arm[1]), int(arm[3])
            odds,co=log_odds(codes[reset,camera],reset,camera)
            map_actions.append(odds>1e-12)
            risk_actions.append(odds>SOURCE_LOG_CUTOFF)
            if not camera:
                assert np.max(np.abs(co)) < 1e-12
            cause_rates.append((co>1e-12).mean(-1))
        always=np.zeros_like(map_actions[0])
        oracle=np.broadcast_to(np.array([True,False])[None,:,None,None],always.shape)
        route=np.stack([*map_actions,*risk_actions,always,~always,oracle],axis=-1)
        rates.append(route.mean(-2).transpose(2,0,1,3)) # material,cause,regime,method
        causes.append(np.stack(cause_rates,axis=-1).transpose(2,0,1,3))
        digests.append([hashlib.sha256(x.read_bytes()).hexdigest() for x in (score_file,pilot_file)])
    return tuple(map(np.asarray,(fine,ultra,real,rates,gates,digests,causes)))


def risk_by_catalog(exact,rate):
    target=np.einsum('cfea,ze->cfza',exact,MIX)
    return target[...,0,None]+rate*(target[...,1]-target[...,0])[...,None]


def weights(rate):
    return np.einsum('cfzma,ze,d->mcfeda',np.stack([1-rate,rate],axis=-1),MIX,EVENT)/4


def contrasts(samples,adjust=4,pairs=COMPARISONS):
    out={}
    for a,b in pairs:
        delta=samples[:,a]-samples[:,b]
        ci=np.quantile(delta[1:],[.025/adjust,1-.025/adjust])
        out[f'{METHODS[a]}-{METHODS[b]}']=[delta[0],*ci]
    return out


def evaluation(fine,ultra,raw,rate,w,real_by_e,overall):
    brier=risk_by_catalog(fine,rate)
    means=w@brier.mean((1,2))
    acc=w@((rate[:,:,0]+1-rate[:,:,1]).mean(1)/2)
    endpoint_weights=weights(rate)
    tail=cdf_boot(raw[0],endpoint_weights,w)
    report=dict(mean=means[0],session_p90=tail[0],accuracy=acc[0],fixed_by_regime=rate.mean((0,1)),
                mean_by_regime=brier.mean((0,1)),contrasts={})
    for key,samples in [('mean',means),('accuracy',acc),('p90',tail)]:
        report['contrasts'][key]=contrasts(samples,4 if overall else 1)
    if overall:
        ultra_mean=w@risk_by_catalog(ultra,rate).mean((1,2))
        replicas=[w@risk_by_catalog(y,rate).mean((1,2)) for y in real_by_e]
        extras=[(4,5),(6,7),(4,6),(5,7),(0,8),(0,9),(4,8),(4,9)]
        report['pointwise']={k:contrasts(x,1,COMPARISONS+extras) for k,x in [('mean',means),('p90',tail),('accuracy',acc)]}
        report['primary_interaction']={}
        for k,x in [('mean',means),('p90',tail),('accuracy',acc)]:
            interaction=x[:,0]-x[:,1]-x[:,2]+x[:,3]
            report['primary_interaction'][k]=[interaction[0],*np.quantile(interaction[1:],[.025,.975])]
        report.update(ultra_means=ultra_mean[0],realized_means=[x[0] for x in replicas],
                      robustness_deltas=[contrasts(x) for x in [ultra_mean,*replicas]],
                      second_p90=cdf_boot(raw[1],endpoint_weights,w[:1])[0])
        # Validate empirical inverse CDF at the reported p90 without pooling object-level losses.
        for k in range(len(ARMS)):
            endpoint=raw[0]
            wt=endpoint_weights[k][...,None]/(C*endpoint.shape[-1])
            val=tail[0,k]
            left=np.sum(wt*(endpoint<val)); right=np.sum(wt*(endpoint<=val))
            assert left<=.9+1e-9 and right>=.9-1e-9
    return report


def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for line in (ROOT/'score_freeze59.sha256').read_text().splitlines():
        sha,name=line.split(maxsplit=1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    fine,ultra,real,rates,gates,digests,cause_rates=read_panels()
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    real_by_e=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    picks=np.random.default_rng(59096657).integers(C,size=(5000,C))
    w=np.vstack([np.full((1,C),1/C),np.stack([(picks==i).mean(1) for i in range(C)],axis=1)])
    avg=rates.mean(2)
    report=evaluation(fine,ultra,raw,avg,w,real_by_e,True)
    by_cause={k:evaluation(fine,ultra,raw,rates[:,:,j],w,real_by_e,False) for j,k in enumerate(['physical','common'])}
    gap=np.einsum('cfea,ze->cfza',fine,MIX)
    gap=gap[...,0]-gap[...,1]
    cacc=(cause_rates[:,:,0]+1-cause_rates[:,:,1]).mean((1,2))/2
    result=dict(methods=METHODS,primary=report,by_cause=by_cause,source_mapping_correct=[int((gap[...,0]>0).sum()),int((gap[...,1]<0).sum())],
                gap_gate_minus_fixed=gap.mean((0,1)),final_deployment_impact_rate=(gates[:,:,None,None]+(1-gates[:,:,None,None])*avg).mean((0,1,2)),
                cause_accuracy=cacc.mean(0),cause_accuracy_contrasts=contrasts(w@cacc))
    np.savez_compressed(ROOT/'score59_cells.npz',route=rates,exact_by_e=fine,ultra_by_e=ultra,gate_purchase=gates,
                        realized_by_e=real_by_e,methods=METHODS,input_sha256=digests,frac_class_physical=cause_rates)
    (ROOT/'score59.json').write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    print('Completed score;',report['accuracy'],report['mean'],report['session_p90'])
if __name__=='__main__':
    main()
