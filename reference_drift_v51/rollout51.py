"""Paired, fully realized deployment audit. Each trial uses a single binary solver outcome."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t
from world51 import EPSILON, DRIFT, KAPPA, paired_calibration_rows, full_states, marginal_retention

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'polarity_calibration_v50'
sys.path.insert(0,str(SOURCE))
from estimator50 import infer, DEAD_ZONE

METHODS=('likelihood_soft','vote_soft','unsigned_fit','vote_hard')
NREP=250
BASE=510317129


def select(lab,idx,state):
    f,h,rad,offset,e,obj=idx
    return lab[f,h,state,rad,offset,e,obj]


def interval(diff,ncontrasts=1):
    mean=np.mean(diff,axis=0)
    half=np.std(diff,axis=0,ddof=1)*t.ppf(1-.025/ncontrasts,len(diff)-1)/np.sqrt(len(diff))
    return np.stack((mean,mean-half,mean+half),axis=-1)


def panel(i):
    g=np.random.default_rng(np.random.SeedSequence([BASE,i]))
    n=f'panel_{i:02d}.npz'
    with np.load(SOURCE/'forecast50'/n) as fz, np.load(SOURCE/'labels50'/n) as lz, np.load(SOURCE/'worlds'/n) as wz, np.load(SOURCE/'cal50'/n) as cz:
        partner=paired_calibration_rows(wz['surface'],wz['law'])
        assert int(wz['support_panel']) != int(wz['cal_support_panel'])
        p=fz['p']; names=list(fz['methods'].astype(str)); labels=lz['labels']
        weak,full=cz['weak_mm'],cz['full_mm']
        kpost_archive=cz['khat']
    layout=(4,NREP,128)
    # All 128 deployed specimens are used exactly once per session.
    obj=np.arange(128)[None,None,:]
    env=np.arange(4)[:,None,None]
    f,h,a,v,rad,off,noise=[g.integers(size,size=layout) for size in (4,2,2,2,2,3,4)]
    weak_state=g.choice(3,p=(.5,.25,.25),size=layout)
    active=weak_state!=0
    each=np.array([g.permutation(np.repeat(np.arange(5),NREP//5)) for _ in range(4)])
    k=np.array(KAPPA)[each]
    rep=g.integers(256,size=(4,NREP))
    ci=(each,partner[:,None],rep)
    pair_w,pair_f=weak[ci][...,:4],full[ci][...,:4]
    alpha=infer(pair_w,pair_f)
    np.testing.assert_allclose(alpha,kpost_archive[each,2,partner[:,None],rep],atol=2e-15,rtol=0)
    readable=(abs(pair_w)>=DEAD_ZONE[0])&(abs(pair_f)>=DEAD_ZONE[1])
    vote=(1+(readable&((pair_w>0)==(pair_f>0))).sum(-1))/(2+readable.sum(-1))
    ix=(f,h,weak_state,a,v,env,obj,noise)
    lp,lm,unsigned=[p[names.index(key)][ix] for key in ('local_plus','local_minus','local_unsigned_fit')]
    pred=np.stack([lm+(lp-lm)*alpha[...,None],lm+(lp-lm)*vote[...,None],unsigned,np.where(vote[...,None]>=.5,lp,lm)])
    # One calibration prefix and one fixed forecast for all 36 interventions.
    snap=pred.copy()
    uref,uflip=g.random((2,*layout)); udrift=g.random((4,NREP))
    indx=(f,h,rad,off,env,obj)
    y_same=select(labels,indx,weak_state)
    y_opp=select(labels,indx,np.where(active,3-weak_state,0))
    sessions=np.empty((6,6,4,NREP,len(METHODS)))
    measures=np.empty((6,6,8))
    # agreement with analytic q marginals, by calibration kappa
    retention=np.empty((6,6,5))
    for j,epsilon in enumerate(EPSILON):
        for d,drift in enumerate(DRIFT):
            ref,impact,shifted=full_states(weak_state,k,uref,udrift,uflip,epsilon,drift)
            y=np.where(impact==weak_state,y_same,y_opp)
            yref=np.where(ref==weak_state,y_same,y_opp)
            if i==0:
                np.testing.assert_array_equal(y,select(labels,indx,impact))
            assert np.all(impact[~active]==0) and np.all(ref[~active]==0)
            sessions[j,d]=((pred-y[None])**2).mean(-1).transpose(1,2,0)
            measures[j,d]=[shifted.mean(),(ref[active]!=impact[active]).mean(),(weak_state[active]==impact[active]).mean(),
                            (y!=yref).mean(),(y_opp[active]!=y_same[active]).mean(), y.mean(), active.mean(), (yref!=y_same).mean()]
            for c in range(5):
                sel=(each==c)[...,None]&active
                retention[j,d,c]=np.mean(weak_state[sel]==impact[sel])
    np.testing.assert_array_equal(sessions[0,0],sessions[-1,-1])
    np.testing.assert_array_equal(pred,snap)
    return sessions,measures,retention,each,partner


def session_metrics(values,kindex):
    # Remove differences between materials and true pre-replacement kappa first.
    # Shape catalogs, env, replicate, two policy contrasts.
    output=np.zeros((len(values),2))
    for i in range(len(values)):
        for e in range(4):
            for c in range(5):
                sample=values[i,e,kindex[i,e]==c]
                output[i]+=sample.var(0,ddof=1)/20
    return output


def bootstrap(m1,m2,boots):
    point=np.sqrt(m1.mean(0))-np.sqrt(m2.mean(0))
    draw=np.sqrt(m1[boots].mean(1))-np.sqrt(m2[boots].mean(1))
    return np.concatenate((point[:,None],np.quantile(draw,(.025,.975),axis=0).T),axis=1)


def main():
    assert os.getenv('SLURM_JOB_ID')
    records=[panel(i) for i in range(12)]
    vals=np.stack([r[0] for r in records]); audit=np.stack([r[1] for r in records])
    ret=np.stack([r[2] for r in records]).mean(0); kid=np.stack([r[3] for r in records])
    partners=np.stack([r[4] for r in records]); np.testing.assert_array_equal(partners,np.tile([2,3,0,1],(12,1)))
    model=np.array([[[marginal_retention(c,e,d) for c in KAPPA] for d in DRIFT] for e in EPSILON])
    largest=float(abs(ret-model).max())
    # paired across common catalog/material, kappa and prefix draws.
    delta=vals[...,0,None]-vals[...,1:3]
    fleet=delta.mean((0,3))  # (eps, drift, rep, comparator)
    pair_keys=((EPSILON.index(.25),0),(0,DRIFT.index(.25)))
    share=[]; residual=[]
    for j,d in pair_keys:
        s=delta[:,j,d] # catalog, env, rep, comparator
        var=session_metrics(s,kid); residual.append(var)
        share.append(dict(session_mean=s.mean((0,1,2)),within_session_sd=np.sqrt(var.mean(0)),
                          pooled_session_p90=np.quantile(s,.9,axis=(0,1,2)),session_worse_rate=(s>0).mean((0,1,2)),
                          fleet_worse_rate=(fleet[j,d]>0).mean(0),fleet_p90=np.quantile(fleet[j,d],.90,axis=0),
                          fleet_rep0=s[:,:,0].mean((0,1)),fleet_rep0_ci=interval(s[:,:,0].mean(1),2)))
    boots=np.random.default_rng(BASE).integers(12,size=(10000,12))
    sd_difference=bootstrap(residual[1],residual[0],boots)
    average=vals.mean((0,3,4))
    fleet0=vals[:,:,:,:,0].mean(3)
    out=dict(epsilon=EPSILON,drift=DRIFT,kappa=KAPPA,methods=METHODS,
             observations='12 paired catalogs x 4 actual opposite-surface sessions x 128 objects, 250 repeated simulated fleets. Per environment: 50 replicates per presurface kappa; conditional repeated draws are not new materials.',
             archive_object_count=6144,method_order=METHODS,comparators=['vote_soft','unsigned_fit'],
             average_loss=average,one_fleet=interval(fleet0),
             first_fleet_soft_minus_baselines=interval(fleet0[:,:,:,0,None]-fleet0[:,:,:,1:3],2),
             fleet_any_worse_rate=(fleet>0).mean(2),grid_audits=audit.mean(0),
             grid_audits_columns=['drift_event_frequency','ref_impact_disagreement_given_active','weak_impact_agreement_given_active',
             'label_changes_ref_vs_impact_all','label_changes_if_sign_flipped_active','impact_stick_all','active_frequency','label_changes_weak_vs_ref_all'],
             measured_retention_by_pre_k=ret,max_abs_retention_vs_formula=largest,
             surface_rows=partners,matched_cells=pair_keys,
             matched_cell_keys=['object_e=.25,g=0','session_e=0,g=.25'],paired_tail_metrics=share,
             sd_session_drift_minus_object_disagreement=sd_difference,
             matched_fleet_Brier_diff=interval(delta[:,0,2].mean((1,2))-delta[:,2,0].mean((1,2))),
             max_abs_drift_frequency=float(abs(audit.mean(0)[:,:,0]-np.array(DRIFT)[None,:]).max()),
             max_abs_ref_error_frequency=float(abs(audit.mean(0)[:,:,1]-np.array(EPSILON)[:,None]).max()))
    np.savez_compressed(ROOT/'rollout_sessions51.npz',risk=vals,pre_k=kid,source_environment=partners,method=METHODS)
    (ROOT/'rollout51.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
    print('max retention deviation',largest,'SD contrast',sd_difference,flush=True)

if __name__=='__main__':
    main()
