"""Compare video readout coverage to the complete design of 24 title cards."""
import argparse
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def records(p):
    with p.open() as f:return list(csv.DictReader(f))
def coverage(track_file):
    cards=records(root/'audit'/'preston'/'validation'/'title_inventory.csv')
    used={int(r['id']):r for r in records(track_file)}
    assert len(cards)==24 and set(used).issubset({int(c['image_ref']) for c in cards})
    interval={int(r['id']):r for r in records(root/'audit'/'preston'/'sign_sensitivity.csv')}
    assert set(used)==set(interval)
    counts={}
    for c in cards:
        i=int(c['image_ref']);r=used.get(i)
        for field,value in (('speed',c['speed_card']),('spin',c['incoming_spin_card']),
               ('setup',c['incoming_spin_card']+' / '+c['speed_card']),('impact_height',c['impact_blade_third']),
               ('prior_bounce',c['prior_bounce_card'])):
            dest=counts.setdefault(field,{}).setdefault(value,{'total':0,'retained':0,'sign_resolved':0})
            dest['total']+=1
            if r:
                dest['retained']+=1
                bounds=interval[i]
                if float(bounds['min_out'])>0 or float(bounds['max_out'])<0:dest['sign_resolved']+=1
    return {'source':str(track_file.relative_to(root)),'count':len(used),
            'sign_validity':'outgoing interval excludes zero in sign_sensitivity.csv','groups':counts}
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('csv',type=Path)
    p=ap.parse_args().csv.resolve()
    (root/'analysis'/'anti_coverage.json').write_text(json.dumps(coverage(p),indent=2)+'\n')
