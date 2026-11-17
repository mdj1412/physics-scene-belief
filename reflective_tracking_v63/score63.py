"""Evaluate loss of each sampled action, integrating the count-matched policy exactly."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'exposure_abstention_v62'))
from thinning62 import split_codes, classify, randomized
from score62 import (C,B,SOURCE_LOG_CUTOFF,MIX,EVENT,weights,cdf_boot,make_weights,target_assign,
                     components,averages,contrast,paired_session_tail,digest)

METHODS=['confidence','random_k_of_n','v54_gate','v61_reset','electronics_reset','fixed_3R_I']
CELLS=[(0,0),(.35,0),(0,.35),(.35,.35)]
COMPARATORS=(1,2)

def board(x):
    # Preserve material, physical/common cause and hazard regime order for deployment join.
    assert x.shape[:4]==(4,2,2,2) and x.shape[4]==256
    return x.reshape(4,2,2,2,-1,B,x.shape[-1]).mean(4).transpose(4,0,3,1,2,5)

def load_pilot(i):
    p=ROOT/'pilots'/f'panel_{i:02d}.npz'
    with np.load(p) as f:
        a=f['codes']
        assert a.shape==(4,3,2,2,2,2,256,4) and a.max()<169
        np.testing.assert_array_equal(f['board']%B,np.arange(256)%B)
    orig=split_codes(a[:,1,1]).reshape(-1,8)
    conf=split_codes(a[:,2,1]).reshape(-1,8)
    base=np.where(orig>0,1+(orig.astype(int)-1)%4,0).astype('u1')
    accepted=orig>=5
    kept=conf>=5
    n,k=accepted.sum(1),kept.sum(1)
    assert np.all(~kept | accepted)
    np.testing.assert_array_equal(conf,np.where(kept,orig,base))
    full=classify(orig)
    chosen=classify(conf)
    random=randomized(orig,base,accepted,kept)
    el=classify(base)
    mask=(k>0)&(k<n)
    np.testing.assert_array_equal(chosen[~mask],random[~mask])
    assert np.all((random>=0)&(random<=1))
    mean_route=np.stack([chosen,random,np.zeros_like(chosen),full,el,np.ones_like(chosen)],axis=1)
    # opt-admissible matching extracts which sample, independent of downstream labels.
    action_diff=chosen[:,0].astype(float)-random[:,0]
    diag_diff=chosen[:,1].astype(float)-random[:,1]
    extra=np.stack([mask,action_diff,abs(action_diff),diag_diff,abs(diag_diff),k/8,
                    (orig>0).mean(1),n/8,n>0,k>0,chosen[:,0]*(1-random[:,0]),
                    (1-chosen[:,0])*random[:,0]],axis=-1)
    shape=(4,2,2,2,256,-1)
    return board(mean_route[...,0].reshape(shape)),board(mean_route[...,1].reshape(shape)),board(extra.reshape(shape)),digest(p)

def load_deployment(i):
    p=ROOT/'scored63'/f'panel_{i:02d}.npz'
    material=((np.arange(4)+i)%4)//2
    with np.load(p) as f:
        pair=[np.stack([f[key][:,:,material==s,:2].mean((1,2)) for s in (0,1)])
              for key in ('exact_cells','ultra_cells')]
        gate=[f['gate'][:,material==s].mean() for s in (0,1)]
        realized=f['real'][...,:2]
        assert realized.shape==(2,3,2,5,4,256,2)
        replay=np.stack([np.take(realized,np.flatnonzero(material==s),axis=4).reshape(2,3,2,-1,2)
                         for s in (0,1)],axis=1)
    return *pair,gate,replay,digest(p)

def interval(arr,n=1):
    return np.r_[arr[0],np.quantile(arr[1:],[.025/n,1-.025/n])]

def risks(fine,route,cause,prod):
    return averages(components(fine,route,cause),prod)

def diffs(value,n=2):
    return [contrast(value,0,j,n) for j in COMPARATORS]

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
    for l in (ROOT/'score_freeze63.sha256').read_text().splitlines():
        h,t=l.split(maxsplit=1)
        assert digest(ROOT/t)==h
    rates,causes,extra,ph=map(np.stack,zip(*(load_pilot(i) for i in range(C))))
    fine,ultra,gates,real,lh=map(np.stack,zip(*(load_deployment(i) for i in range(C))))
    cat,board_w,prod=make_weights(6000,6309721)
    rel='independent'
    r=target_assign(rates,rel);c=target_assign(causes,rel);s=target_assign(extra,rel)
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    replay=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    bank=np.einsum('cfea,ze->cfza',fine,MIX);gap=bank[...,1]-bank[...,0]
    def strat(routes,diagnosis,extras,n):
        vals=risks(fine,routes,diagnosis,prod)
        out={k:v[0] for k,v in vals.items()}
        out['cause_BA']=[None if i in (2,5) else v for i,v in enumerate(out['cause_BA'])]
        out['fixed_rate']=routes.mean((0,1,2,3,4))
        out['mean_Brier_confidence_minus']=diffs(vals['mean_Brier'],n)
        out['lower_loss_action_BA_confidence_minus']=diffs(vals['lower_loss_action_BA'],n)
        out['cause_BA_confidence_minus_random']=contrast(vals['cause_BA'],0,1,1)
        ex=prod@extras.mean((2,3,4)).reshape(C*B,-1)
        share,ds,dsabs,diag,diagabs,paid,active,basepaid,hasv,hasc,tofixed,togate=ex.T
        riskdiff=vals['mean_Brier'][:,0]-vals['mean_Brier'][:,1]
        # Translate changed action into a loss on each surface and target regime before averaging.
        dgap=gap[:,None,:,None,:]
        gen=prod@(extras[...,1]*dgap).mean((2,3,4)).reshape(C*B)
        assert np.max(np.abs(gen-riskdiff))<4e-15
        assert np.allclose(dsabs,tofixed+togate) and np.all(dsabs<=share+1e-14)
        out['selection']={
            'flexible_pilot_share':interval(share),
            'action_changed_probability':interval(dsabs),
            'confidence_fixed_random_gate':interval(tofixed),'confidence_gate_random_fixed':interval(togate),
            'cause_changed_probability':interval(diagabs),
            'conditional_mean_Brier_diff_on_flexible':interval(gen/share),
            'cond_action_change_on_flexible':interval(dsabs/share),
            'active_coverage_conf_and_random':interval(paid/active),'active_coverage_original':interval(basepaid/active),
            'paid_coverage_conf_and_random':interval(paid),'paid_coverage_original':interval(basepaid),
            'has_any_conf_random':interval(hasc),'has_any_original':interval(hasv)
        }
        return out,vals['mean_Brier']
    out=dict(methods=METHODS,cells=CELLS,catalogs=C,boards=B,episodes_per_board_cell=32)
    route=r.mean(2);cause=c.mean(2)
    out['uniform'],val=strat(route,cause,s.mean(2),2)
    cells=[strat(r[:,:,i],c[:,:,i],s[:,:,i],8) for i in range(4)]
    out['per_cell']=[o for o,v in cells]
    cell_risk=np.stack([v[:,0,None]-v[:,COMPARATORS] for o,v in cells])
    out['factorial_diff_in_risk_contrast']={name:[interval(a[:,i]) for i in range(2)] for name,a in
        [('transient',((cell_risk[1]-cell_risk[0])+(cell_risk[3]-cell_risk[2]))/2),
         ('coherent',((cell_risk[2]-cell_risk[0])+(cell_risk[3]-cell_risk[1]))/2),
         ('interaction',cell_risk[3]-cell_risk[2]-cell_risk[1]+cell_risk[0])]}
    rate=route.mean(3)
    w=weights(rate.mean(1))
    for seed in (0,1):
        p90=cdf_boot(raw[seed],w,cat[:1])[0]
        out['uniform'][f'replay{seed+1}_session_p90']=p90
        for j,v in enumerate(p90):
            pm=w[j][...,None]/(C*raw.shape[-1]);q=raw[seed]
            assert (pm*(q<v)).sum()<=.9+1e-9 and (pm*(q<=v)).sum()>=.9-1e-9
    tails=np.stack([paired_session_tail(raw[0],rate,cat,board_w,j) for j in (0,1,2)],-1)
    assert np.max(abs(tails[0]-out['uniform']['replay1_session_p90'][:3]))<1e-8
    out['uniform']['p90_confidence_minus']=diffs(tails)
    out['sensitivity']={name:diffs(risks(y,route,cause,prod)['mean_Brier']) for name,y in
                       zip(['ultra','binary1','binary2'],[ultra,*replay])}
    out['source_relations']={}
    for t in ('matched','reversed'):
        ar=target_assign(rates,t).mean(2);ac=target_assign(causes,t).mean(2)
        out['source_relations'][t]=diffs(risks(fine,ar,ac,prod)['mean_Brier'])
    out['oracle_diagnostic']={'mean_gate_minus_fixed_by_regime':-gap.mean((0,1)),
                              'A_fixed_B_gate_counts':[(gap[...,0]<0).sum(),(gap[...,1]>0).sum()],
                              'surface_regimes_per_count':int(np.prod(gap.shape[:-1])),
                              'clairvoyant_risk':bank.min(-1).mean()}
    wb=np.tile(np.eye(B)[:,None,:]/C,(1,C,1)).reshape(B,C*B)
    ob=risks(fine,route,cause,wb)['mean_Brier']
    out['by_board_conf_minus_random_and_gate']=ob[:,0,None]-ob[:,COMPARATORS]
    np.savez_compressed(ROOT/'score63_cells.npz',rates=rates,causes=causes,extras=extra,
                        fine=fine,ultra=ultra,replay=replay,gates=gates,pilot_hash=ph,
                        forecast_hash=lh,methods=METHODS)
    (ROOT/'score63.json').write_text(json.dumps(out,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')

if __name__=='__main__':
    main()
