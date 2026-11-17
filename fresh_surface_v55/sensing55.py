"""Surface replacement sensing costs; evaluated on prefixes before looking at labels."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'adaptive_witness_v54'))
from policy54 import select54

def main():
    counts=[]
    totals=[]
    for i in range(24):
        with np.load(ROOT/'raw'/f'panel_{i:02d}.npz') as a:
            r=select54(a['nw'][...,:3],a['nf'][...,:3],mode='robust')
            counts.append(int(r.sum()))
            totals.append(r.size)
    p=sum(counts)/sum(totals)
    rows=[]
    for t in (.52,1.04,2.08):
        for name,r in [('source_risk_gate',p),('fixed_3RI',1.),('rate_matched_random',p)]:
            rows.append(dict(impact_and_reset_s=t,arm=name,weak=4,reference=4-r,impact=r,
                             contacts=8,resets=8,seconds_per_session=5.92+r*(t-1.04),
                             ms_per_deployment_object=(5.92+r*(t-1.04))/128*1000))
    data=dict(impact_counts_per_catalog=counts,total_prefixes_per_catalog=totals,
              gate_impact_fraction=p,weak_and_reset_s=.44,reference_and_reset_s=1.04,
              modeled_cost=rows,old_contacts_common=8,old_resets_common=8)
    (ROOT/'sensing55.json').write_text(json.dumps(data,indent=2))
    print(sum(counts),sum(totals),p)
if __name__=='__main__':main()
