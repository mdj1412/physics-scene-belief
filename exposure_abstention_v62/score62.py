"""Frozen acquisition choices and deployment risk on the crossed pulse-by-exposure panel."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'switch_diode_v61'))
from score61 import (C,B,MIX,EVENT,SOURCE_LOG_CUTOFF,log_odds,digest,weights,cdf_boot,
                     make_weights,target_assign,components,averages,contrast,paired_session_tail)

ARMS = ((2,1),(1,1),(1,0),(2,0),(0,1),(0,0))
METHODS = ('selective_reset','v61_reset','v61_sham','selective_sham','nominal_reset',
           'nominal_sham','gate','fixed','selective_map_reset','v61_map_reset')
CELLS = ((.2,.25),(.2,.8),(.8,.25),(.8,.8))
COMPARATORS = (1,2,6)
TARGETS = ('independent','matched','reversed')

def board_average(val):
    # Cell, mechanism, regime, material, episodes -> board, cell, material, mechanism, regime
    return val.reshape(4,2,2,2,-1,B).mean(-2).transpose(4,0,3,1,2)

def load_pilot(c):
    p=ROOT/'pilots'/f'panel_{c:02d}.npz'
    with np.load(p) as f:
        codes=f['codes']
        assert codes.shape==(4,3,2,2,2,2,512,4) and codes.max()<169
        key='board_id' if 'board_id' in f else 'board'
        assert np.array_equal(np.arange(512)%B,f[key]%B)
    acts=[]; maps=[]; causes=[]; paid=[]
    for mode,reset in ARMS:
        code=codes[:,mode,reset]
        z,k=log_odds(code,reset,True)
        acts.append(z>SOURCE_LOG_CUTOFF)
        maps.append(board_average(z>1e-12))
        causes.append(board_average(k>1e-12))
        # Every paid object has one reference-impact pair. Symbols with an observed camera
        # sign-pair have a status digit >=1 in the camera dimension.
        sym=np.stack([code//13,code%13],axis=-1)
        paid.append(np.stack([board_average((sym>0).mean((-2,-1))),
                              board_average((sym>=5).mean((-2,-1)))],axis=-1))
    for reset in range(2):
        a,b=codes[:,2,reset],codes[:,1,reset]
        for aa,bb in ((a//13,b//13),(a%13,b%13)):
            assert np.all((aa==bb)|((aa>0)&(aa<=4)&(bb>=5)))
    zero=np.zeros_like(maps[0])
    rate=np.stack([*[board_average(x) for x in acts],zero,1-zero,maps[0],maps[1]],axis=-1)
    switches=[]
    for comparator in (acts[1],acts[2],np.zeros_like(acts[0])):
        switches.extend([board_average((~acts[0])&comparator),
                         board_average(acts[0]&(~comparator))])
    return rate,np.stack(causes,axis=-1),np.stack(switches,axis=-1),np.stack(paid,axis=-2),digest(p)

def load_deployment(c):
    p=ROOT/'scored62'/f'panel_{c:02d}.npz'
    material=((np.arange(4)+c)%4)//2
    with np.load(p) as f:
        pair=[np.stack([f[key][:,:,material==s,:2].mean((1,2)) for s in (0,1)])
              for key in ('exact_cells','ultra_cells')]
        gate=[f['gate'][:,material==s].mean() for s in (0,1)]
        ar=f['real'][...,:2]
        assert ar.shape==(2,3,2,5,4,256,2)
        replay=np.stack([np.take(ar,np.flatnonzero(material==s),axis=4).reshape(2,3,2,-1,2)
                        for s in (0,1)],axis=1)
    return *pair,gate,replay,digest(p)

def joint(squared,route,causes,raw,prod,ncontrast):
    vals=averages(components(squared,route,causes),prod)
    r=route.mean(3)
    point={k:v[0] for k,v in vals.items()}
    w=weights(r.mean(1))
    point['session_p90']=cdf_boot(raw,w,np.full((1,C),1/C))[0]
    out={}
    for k,v in vals.items():
        n=(2 if ncontrast==3 else ncontrast*2//3) if k=='cause_BA' else ncontrast
        out[k]=[contrast(v,0,j,n) if k!='cause_BA' or j!=6 else None for j in COMPARATORS]
    point['vs_v61reset_v61sham_gate']=out
    point['fixed_rate_AB']=r.mean((0,1,2)); point['fixed_by_cause_AB']=route.mean((0,1,2))
    return point

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for row in (ROOT/'score_freeze62.sha256').read_text().splitlines():
        h,path=row.split(maxsplit=1)
        assert digest(ROOT/path)==h
    allpilot=[load_pilot(c) for c in range(C)]
    rates,cause,switch,paid,ph=map(np.asarray,zip(*allpilot))
    fine,ultra,gate,real,lh=map(np.asarray,zip(*[load_deployment(c) for c in range(C)]))
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    replay=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    cat,board,product=make_weights(6000,6204118)
    report=dict(methods=METHODS,optics_ms=CELLS,readers=['nominal','v61','selective'],
                catalogs=C,boards=B,episodes_per_board=64,uniform={},cells={})
    counts=paid.mean((0,1,3,4,5))
    report['paired_camera_coverage_by_cell']=counts[:,:,1]/counts[:,:,0]
    report['paid_camera_coverage_by_cell']=counts[:,:,1]
    report['paired_camera_coverage_total']=counts.mean(0)[:,1]/counts.mean(0)[:,0]
    report['paid_camera_coverage_total']=counts.mean(0)[:,1]
    for relation in TARGETS:
        a=target_assign(rates,relation);k=target_assign(cause,relation)
        report['cells'][relation]=[joint(fine,a[:,:,d],k[:,:,d],raw[0],product,36) for d in range(4)]
        v=a.mean(2);co=k.mean(2);r=v.mean(3)
        summary=joint(fine,v,co,raw[0],product,3)
        summary['deployment_final_impact_rate']=(gate[:,None,:,None,None]+(1-gate[:,None,:,None,None])*r).mean((0,1,2,3))
        if relation=='independent':
            parts=components(fine,v,co)
            wb=np.tile(np.eye(B)[:,None,:]/C,(1,C,1)).reshape(B,C*B)
            wo=np.tile((1-np.eye(B))[:,None,:]/((B-1)*C),(1,C,1)).reshape(B,C*B)
            summary['by_board']=averages(parts,wb)
            summary['leave_one_board_out']=averages(parts,wo)
            inds=[0,*COMPARATORS]
            tail=np.stack([paired_session_tail(raw[0],r,cat,board,i) for i in inds],-1)
            assert np.max(abs(tail[0]-summary['session_p90'][inds]))<1e-8
            summary['vs_v61reset_v61sham_gate']['session_p90']=[contrast(tail,0,j,3) for j in (1,2,3)]
            w=weights(r.mean(1))
            summary['seed2_p90']=cdf_boot(raw[1],w,np.full((1,C),1/C))[0]
            for j in inds:
                q=summary['session_p90'][j]
                wt=w[j][...,None]/(C*raw.shape[-1])
                assert (wt*(raw[0]<q)).sum()<=.9+1e-10 and (wt*(raw[0]<=q)).sum()>=.9-1e-10
            checks={}
            for label,y in zip(('ultra','binary1','binary2'),[ultra,*replay]):
                means=averages(components(y,v,co),product)['mean_Brier']
                checks[label]=[contrast(means,0,j,3) for j in COMPARATORS]
            summary['robustness']=checks
        # Resolve directional loss: selective gate vs comparator fixed, and vice versa.
        sw=target_assign(switch,relation).mean((2,4)) # C,B,F,Z,6
        gap=np.einsum('cfea,ze->cfza',fine,MIX)
        diff=gap[...,1]-gap[...,0]
        losses=(sw*diff[:,None,:,:,None]*np.tile([-1,1],3)).mean((0,1,2,3)).reshape(3,2)
        switches=sw.mean((0,1,2,3)).reshape(3,2)
        assert np.max(abs(losses.sum(-1)-(summary['mean_Brier'][0]-summary['mean_Brier'][list(COMPARATORS)])))<1e-13
        summary['changes_to_gate_to_fixed']=switches
        summary['loss_cost_to_gate_to_fixed']=losses
        report['uniform'][relation]=summary
        print('completed',relation,flush=True)
    bank=np.einsum('cfea,ze->cfza',fine,MIX)
    gaps=bank[...,0]-bank[...,1]
    report['gate_minus_fixed_AB']=gaps.mean((0,1))
    report['A_fixed_B_gate_counts']=[np.sum(gaps[...,0]>0),np.sum(gaps[...,1]<0)]
    report['oracle_surface_risk']=bank.min(-1).mean()
    np.savez_compressed(ROOT/'score62_cells.npz',rates=rates,cause=cause,switches=switch,paid=paid,
                        fine=fine,ultra=ultra,replay=replay,gates=gate,pilot_hash=ph,forecast_hash=lh,
                        methods=METHODS,cells=CELLS)
    (ROOT/'score62.json').write_text(json.dumps(report,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
if __name__=='__main__': main()
