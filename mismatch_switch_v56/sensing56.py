"""Physical sensing budget from purchased prefix and pilot observations, without labels."""
import json
from pathlib import Path
import sys
import numpy as np
from switch56 import choose, W, PRIMARY, BUDGETS

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'adaptive_witness_v54'))
from policy54 import select54

def physical_cost(k,h,rate,t):
    # h downstream sessions on one surface reuse a single set of k pilot pairs.
    common_weak=4*.44
    common_ref=3*1.04
    slots={ 'pilot_seconds':k*(1.04+t),
            'pilot_contacts':2*k,'pilot_resets':2*k,
            'contacts_per_deployment_including_pilot':8+2*k/h,
            'resets_per_deployment_including_pilot':8+2*k/h,
            'unpaid_padding_per_deployment_s':(1-rate)*max(0,t-1.04)+rate*max(0,1.04-t),
            'active_per_deployment_s':k*(1.04+t)/h+common_weak+common_ref+(1-rate)*1.04+rate*t,
            'equal_slot_per_deployment_s':k*(1.04+t)/h+common_weak+common_ref+max(1.04,t)}
    assert np.isclose(slots['unpaid_padding_per_deployment_s']+slots['active_per_deployment_s'],
                      slots['equal_slot_per_deployment_s'],rtol=0,atol=1e-12)
    return slots

def main():
    by_catalog=[]
    for i in range(24):
        env=(i+np.arange(4))%4//2
        with np.load(ROOT/'raw'/f'panel_{i:02d}.npz') as x:
            g=select54(x['nw'][...,:3],x['nf'][...,:3],mode='robust').mean(axis=(0,2))
        gate=np.array([g[env==m].mean() for m in range(2)])[:,None]
        with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as x:
            ref,imp=x['pilot_ref'],x['pilot_imp']
        stats=[]
        for k in BUDGETS:
            switch=choose(ref[...,:k],imp[...,:k]).mean(axis=-1)
            r=switch+(1-switch)*gate
            stats.append(np.stack((switch,r,np.broadcast_to(gate,r.shape),np.ones_like(r)),axis=-1))
        by_catalog.append(stats)
    arr=np.stack(by_catalog)
    assert arr.shape==(24,len(BUDGETS),2,len(W),4)
    by_world=arr.mean((0,2)); primary=by_world[:,PRIMARY,:].mean(1)
    tables=[]
    for i,k in enumerate(BUDGETS):
        for a,name in enumerate(('pilot_switch','always_v54_gate','fixed_3RI')):
            rate=primary[i,a+1]
            for h in (1,2,16):
                for t in (.52,1.04,2.08):
                    tables.append(dict(K=k,horizon=h,T=t,method=name,impact_fraction=rate,**physical_cost(k,h,rate,t)))
    out=dict(methods=['outer_fixed_rate','impact_rate_switch','impact_rate_gate','impact_rate_fixed'],
             primary_rates=primary.tolist(),by_w=by_world.tolist(),w=W.tolist(),K=BUDGETS,rows=tables,
             common_pre_exchange_contacts=8,common_pre_exchange_resets=8,common_deployment_weak_contacts_per_object=1)
    (ROOT/'sensing56.json').write_text(json.dumps(out,indent=2))
    print('primary switch rate and I rates',k,primary)
if __name__=='__main__':main()
