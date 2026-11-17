"""Compute unpadded contact cost and matched resource caps directly from sensor-only branches."""
from pathlib import Path
import json,sys
import numpy as np
from policy57 import choose

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'adaptive_witness_v54'))
from policy54 import select54
N=24; K=8
METHODS=('MAP_4x2','MAP_8x1','risk_4x2','always_gate','always_fixed_I')

def times(impact_rate,session_count,horizon,t=1.04,setup=0):
    pilot=K*(1.04+t)
    deployment=4*.44+3*1.04+1.04+(t-1.04)*impact_rate
    occupied=(pilot+session_count*setup)/horizon+deployment
    matched=(pilot+8*setup)/horizon+4*.44+3*1.04+max(1.04,t)
    padding=(8-session_count)*setup/horizon+max(t,1.04)-((1-impact_rate)*1.04+impact_rate*t)
    assert np.isclose(occupied+padding,matched)
    return {'regime_setups':session_count,'contact_resets_per_surface_pilot':16,
            'pilot_s':pilot+session_count*setup,'seconds_per_deployment_session':occupied,
            'matched_cap_s':matched,'padding_s':padding,'H':horizon,'T':t,'S':setup}

def main():
    results=[]
    for c in range(N):
        area=((c+np.arange(4))%4)//2
        with np.load(ROOT/'raw'/f'panel_{c:02d}.npz') as d:
            gate=select54(d['nw'][...,:3],d['nf'][...,:3],mode='robust').mean(axis=(0,2))
        gate=np.array([gate[area==i].mean() for i in range(2)])[:,None]
        with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as d:
            ref,imp=d['pilot_ref'],d['pilot_imp']
        assert ref.shape==imp.shape==(2,2,2,4096,8)
        p4=choose(ref[:,:,1],imp[:,:,1]).mean(-1)
        p8=choose(ref[:,:,0],imp[:,:,0],design='8x1').mean(-1)
        p4r=choose(ref[:,:,1],imp[:,:,1],rule='risk').mean(-1)
        s=np.stack([p4,p8,p4r,np.zeros_like(p4),np.ones_like(p4)])
        I=s+(1-s)*gate
        results.append(np.stack([s,I],axis=-1))
    rates=np.mean(results,axis=(0,2))
    flat=rates.mean(axis=1)
    ledger=[]
    for i,method in enumerate(METHODS):
        # For a constant baseline the pilot result is ignored: cost under either design is valid.
        groups=[4] if i in (0,2) else [8] if i==1 else [4,8]
        for g in groups:
            for h in (1,2,16):
                for t in (.52,1.04,2.08):
                    for setup in (0,1):
                        ledger.append(dict(method=method,**times(flat[i,1],g,h,t,setup)))
    record=dict(methods=METHODS,rate_columns=['outer_select_fixed','deployment_buy_impact'],
                by_regime=rates.tolist(),overall=flat.tolist(),K=8,
                pilot_contacts=16,pilot_resets=16,distinct_pilot_specimens=8,post_cal_contacts=8,
                post_cal_resets=8,old_shared_contacts=8,old_shared_resets=8,
                common_pilot_individual_specimen_handling_events=8,per_session_cost=ledger)
    (ROOT/'sensing57.json').write_text(json.dumps(record,indent=2)+'\n')
    print(METHODS,flat.tolist())

if __name__=='__main__':main()
